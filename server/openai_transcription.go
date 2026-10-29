package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sync/errgroup"
	"shelley.exe.dev/models"
)

const (
	maxTranscriptionUpload    = 25_000_000
	maxTranscriptionErrorBody = 64 << 10
	maxTranscriptionResponse  = 16 << 20
)

type transcriptionResult struct {
	Text            string
	Model           string
	TimestampsModel string
	TimestampsPath  string
}

type transcriptionAPIOptions struct {
	Model                  string
	ResponseFormat         string
	TimestampGranularities []string
}

var (
	textTranscription = transcriptionAPIOptions{
		Model:          models.TranscriptionTextModel,
		ResponseFormat: "json",
	}
	timestampTranscription = transcriptionAPIOptions{
		Model:                  models.TranscriptionTimestampsModel,
		ResponseFormat:         "verbose_json",
		TimestampGranularities: []string{"word", "segment"},
	}
)

type recordingTranscriber interface {
	// Available reports why recordings cannot be transcribed, or nil.
	Available() error
	Transcribe(context.Context, string, string, bool) (transcriptionResult, error)
}

type transcriptionModelProvider interface {
	GetTranscriptionModels(string) ([]models.TranscriptionModel, error)
}

type openAIRecordingTranscriber struct {
	client       *http.Client
	resolveModel func(string) (models.TranscriptionModel, error)
	mediaRun     mediaCommandRunner
}

type transcriptionAPIResponse struct {
	Text string
	Body []byte
}

type transcriptionHTTPError struct {
	StatusCode int
	Status     string
	Message    string
}

type transcriptionModelError struct {
	Model string
	Err   error
}

func (e *transcriptionModelError) Error() string { return e.Err.Error() }
func (e *transcriptionModelError) Unwrap() error { return e.Err }

func withTranscriptionModel(model string, err error) error {
	if err == nil {
		return nil
	}
	return &transcriptionModelError{Model: model, Err: err}
}

func (e *transcriptionHTTPError) Error() string {
	return fmt.Sprintf("OpenAI transcription failed (%s): %s", e.Status, e.Message)
}

// With predictableOnly every route is ignored, as real LLM models are.
func newOpenAIRecordingTranscriber(provider LLMProvider, predictableOnly bool) recordingTranscriber {
	return &openAIRecordingTranscriber{
		client: http.DefaultClient,
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			modelProvider, ok := provider.(transcriptionModelProvider)
			if !ok || predictableOnly {
				return models.TranscriptionModel{}, errTranscriptionModelUnavailable(modelName)
			}
			available, err := modelProvider.GetTranscriptionModels(modelName)
			if err != nil {
				return models.TranscriptionModel{}, fmt.Errorf("load transcription model %q: %w", modelName, err)
			}
			return selectTranscriptionModel(modelName, available)
		},
		mediaRun: runMediaCommand,
	}
}

// selectTranscriptionModel prefers an exe.dev LLM integration (the llm one
// first), then a custom model, then a route derived from OpenAI credentials.
func selectTranscriptionModel(modelName string, available []models.TranscriptionModel) (models.TranscriptionModel, error) {
	var selected *models.TranscriptionModel
	best := 0
	for i := range available {
		if available[i].Model != modelName || available[i].Endpoint == "" {
			continue
		}
		if rank := transcriptionRouteRank(available[i]); rank > best {
			selected, best = &available[i], rank
		}
	}
	if selected == nil {
		return models.TranscriptionModel{}, errTranscriptionModelUnavailable(modelName)
	}
	return *selected, nil
}

func transcriptionRouteRank(route models.TranscriptionModel) int {
	switch parsed, err := url.Parse(route.Endpoint); {
	case route.FromCredentials:
		return 1
	case route.Source == models.SourceCustomLabel:
		return 2
	case err == nil && strings.HasPrefix(parsed.Hostname(), "llm.int."):
		return 4
	default:
		return 3
	}
}

var errNoTranscriptionRoute = errors.New("set OPENAI_API_KEY or connect an exe.dev LLM integration")

func errTranscriptionModelUnavailable(modelName string) error {
	return fmt.Errorf("transcription model %q is not available; %w", modelName, errNoTranscriptionRoute)
}

// Available checks both models: screen and review recordings also need
// whisper-1's word timings.
func (t *openAIRecordingTranscriber) Available() error {
	for _, model := range []string{textTranscription.Model, timestampTranscription.Model} {
		if _, err := t.resolveModel(model); err != nil {
			return err
		}
	}
	return nil
}

func (t *openAIRecordingTranscriber) Transcribe(ctx context.Context, mediaPath, prompt string, timestamps bool) (transcriptionResult, error) {
	uploadPath, cleanup, err := t.prepareUpload(ctx, mediaPath)
	if err != nil {
		return transcriptionResult{}, err
	}
	defer cleanup()

	if !timestamps {
		response, err := t.transcribe(ctx, uploadPath, prompt, textTranscription)
		if err != nil {
			return transcriptionResult{}, withTranscriptionModel(textTranscription.Model, err)
		}
		return transcriptionResult{Text: response.Text, Model: textTranscription.Model}, nil
	}

	var transcriptResponse, timestampResponse transcriptionAPIResponse
	group, groupContext := errgroup.WithContext(ctx)
	group.Go(func() error {
		var err error
		transcriptResponse, err = t.transcribe(groupContext, uploadPath, prompt, textTranscription)
		return withTranscriptionModel(textTranscription.Model, err)
	})
	group.Go(func() error {
		var err error
		timestampResponse, err = t.transcribe(groupContext, uploadPath, prompt, timestampTranscription)
		return withTranscriptionModel(timestampTranscription.Model, err)
	})
	if err := group.Wait(); err != nil {
		return transcriptionResult{}, err
	}
	if err := validateTimestampResponse(timestampResponse.Body); err != nil {
		return transcriptionResult{}, withTranscriptionModel(timestampTranscription.Model, err)
	}

	timestampsPath := mediaPath + ".timestamps.json"
	if err := os.WriteFile(timestampsPath, timestampResponse.Body, 0o600); err != nil {
		return transcriptionResult{}, fmt.Errorf("write transcription timestamps: %w", err)
	}
	return transcriptionResult{
		Text:            transcriptResponse.Text,
		Model:           textTranscription.Model,
		TimestampsModel: timestampTranscription.Model,
		TimestampsPath:  timestampsPath,
	}, nil
}

func (t *openAIRecordingTranscriber) prepareUpload(ctx context.Context, mediaPath string) (string, func(), error) {
	info, err := os.Stat(mediaPath)
	if err != nil {
		return "", func() {}, fmt.Errorf("inspect recording upload: %w", err)
	}
	if info.Size() <= maxTranscriptionUpload {
		return mediaPath, func() {}, nil
	}
	if t.mediaRun == nil {
		return "", func() {}, errors.New("oversized recording preparation requires a media command runner")
	}

	file, err := os.CreateTemp("", "shelley-transcription-*.m4a")
	if err != nil {
		return "", func() {}, fmt.Errorf("create prepared transcription audio: %w", err)
	}
	preparedPath := file.Name()
	if err := file.Close(); err != nil {
		_ = os.Remove(preparedPath)
		return "", func() {}, fmt.Errorf("close prepared transcription audio: %w", err)
	}
	cleanup := func() { _ = os.Remove(preparedPath) }
	if _, err := t.mediaRun(ctx, "ffmpeg", "-y", "-i", mediaPath, "-map", "0:a:0", "-vn", "-c:a", "aac", "-b:a", "64k", preparedPath); err != nil {
		cleanup()
		return "", func() {}, fmt.Errorf("prepare oversized recording audio: %w", err)
	}
	preparedInfo, err := os.Stat(preparedPath)
	if err != nil {
		cleanup()
		return "", func() {}, fmt.Errorf("inspect prepared transcription audio: %w", err)
	}
	if preparedInfo.Size() > maxTranscriptionUpload {
		cleanup()
		return "", func() {}, errors.New("prepared recording is still larger than 25 MB; split it or use the transcribing-audio skill for chunked transcription")
	}
	return preparedPath, cleanup, nil
}

func validateTimestampResponse(body []byte) error {
	var decoded struct {
		Words    []json.RawMessage `json:"words"`
		Segments []json.RawMessage `json:"segments"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return fmt.Errorf("decode Whisper timestamp response: %w", err)
	}
	if decoded.Words == nil {
		return errors.New(`Whisper timestamp response field "words" must be an array`)
	}
	if decoded.Segments == nil {
		return errors.New(`Whisper timestamp response field "segments" must be an array`)
	}
	return nil
}

func (t *openAIRecordingTranscriber) transcribe(ctx context.Context, mediaPath, prompt string, options transcriptionAPIOptions) (transcriptionAPIResponse, error) {
	model, err := t.resolveModel(options.Model)
	if err != nil {
		return transcriptionAPIResponse{}, err
	}
	if model.WireModel != "" {
		return t.transcribeChatCompletion(ctx, model, mediaPath, prompt, options)
	}
	media, err := os.Open(mediaPath)
	if err != nil {
		return transcriptionAPIResponse{}, fmt.Errorf("open recording: %w", err)
	}
	defer media.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("model", options.Model); err != nil {
		return transcriptionAPIResponse{}, err
	}
	if err := writer.WriteField("response_format", options.ResponseFormat); err != nil {
		return transcriptionAPIResponse{}, err
	}
	for _, granularity := range options.TimestampGranularities {
		if err := writer.WriteField("timestamp_granularities[]", granularity); err != nil {
			return transcriptionAPIResponse{}, err
		}
	}
	if err := writer.WriteField("prompt", prompt); err != nil {
		return transcriptionAPIResponse{}, err
	}
	part, err := writer.CreateFormFile("file", filepath.Base(mediaPath))
	if err != nil {
		return transcriptionAPIResponse{}, err
	}
	if _, err := io.Copy(part, media); err != nil {
		return transcriptionAPIResponse{}, fmt.Errorf("read recording: %w", err)
	}
	if err := writer.Close(); err != nil {
		return transcriptionAPIResponse{}, err
	}

	return t.request(ctx, model, writer.FormDataContentType(), body.Bytes(), options.ResponseFormat != "verbose_json")
}

func (t *openAIRecordingTranscriber) transcribeChatCompletion(ctx context.Context, model models.TranscriptionModel, mediaPath, prompt string, options transcriptionAPIOptions) (transcriptionAPIResponse, error) {
	audioBytes, err := os.ReadFile(mediaPath)
	if err != nil {
		return transcriptionAPIResponse{}, fmt.Errorf("open recording: %w", err)
	}
	format := strings.TrimPrefix(strings.ToLower(filepath.Ext(mediaPath)), ".")
	if format == "" {
		format = "wav"
	}
	userPrompt := prompt + "\n\nOutput only the raw transcript text, with no commentary or formatting."
	if options.ResponseFormat == "verbose_json" {
		userPrompt = prompt + "\n\nReturn a JSON object with keys \"text\" (full transcript string), \"segments\" (array of {\"start\": number, \"end\": number} in seconds), and \"words\" (array of {\"word\": string, \"start\": number, \"end\": number} in seconds)."
	}
	payload := map[string]any{
		"model": model.WireModel,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": userPrompt},
				{"type": "input_audio", "input_audio": map[string]string{
					"data":   base64.StdEncoding.EncodeToString(audioBytes),
					"format": format,
				}},
			},
		}},
	}
	if options.ResponseFormat == "verbose_json" {
		payload["response_format"] = map[string]string{"type": "json_object"}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return transcriptionAPIResponse{}, err
	}
	responseBody, err := t.sendRequest(ctx, model, "application/json", body)
	if err != nil {
		return transcriptionAPIResponse{}, err
	}
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(responseBody, &completion); err != nil {
		return transcriptionAPIResponse{}, fmt.Errorf("decode OpenAI transcription response: %w", err)
	}
	var content string
	if len(completion.Choices) > 0 {
		content = strings.TrimSpace(completion.Choices[0].Message.Content)
	}
	if options.ResponseFormat == "verbose_json" {
		content = stripJSONCodeFence(content)
		var decoded struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(content), &decoded); err != nil {
			return transcriptionAPIResponse{}, fmt.Errorf("decode OpenAI transcription response: %w", err)
		}
		return transcriptionAPIResponse{Text: strings.TrimSpace(decoded.Text), Body: []byte(content)}, nil
	}
	if content == "" {
		return transcriptionAPIResponse{}, errors.New("OpenAI transcription returned an empty transcript")
	}
	encoded, err := json.Marshal(map[string]string{"text": content})
	if err != nil {
		return transcriptionAPIResponse{}, err
	}
	return transcriptionAPIResponse{Text: content, Body: encoded}, nil
}

func stripJSONCodeFence(s string) string {
	if !strings.HasPrefix(s, "```") {
		return s
	}
	if newline := strings.IndexByte(s, '\n'); newline >= 0 {
		s = s[newline+1:]
	}
	s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	return strings.TrimSpace(s)
}

func (t *openAIRecordingTranscriber) request(ctx context.Context, model models.TranscriptionModel, contentType string, body []byte, requireText bool) (transcriptionAPIResponse, error) {
	responseBody, err := t.sendRequest(ctx, model, contentType, body)
	if err != nil {
		return transcriptionAPIResponse{}, err
	}
	var decoded struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return transcriptionAPIResponse{}, fmt.Errorf("decode OpenAI transcription response: %w", err)
	}
	decoded.Text = strings.TrimSpace(decoded.Text)
	if requireText && decoded.Text == "" {
		return transcriptionAPIResponse{}, errors.New("OpenAI transcription returned an empty transcript")
	}
	return transcriptionAPIResponse{Text: decoded.Text, Body: responseBody}, nil
}

func (t *openAIRecordingTranscriber) sendRequest(ctx context.Context, model models.TranscriptionModel, contentType string, body []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, model.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	if model.APIKey != "" && model.APIKey != "implicit" {
		req.Header.Set("Authorization", "Bearer "+model.APIKey)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("OpenAI transcription request: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, maxTranscriptionResponse+1))
	if err != nil {
		return nil, fmt.Errorf("read OpenAI transcription response: %w", err)
	}
	if len(responseBody) > maxTranscriptionResponse {
		return nil, errors.New("OpenAI transcription response exceeded 16 MiB")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		errorBody := responseBody
		if len(errorBody) > maxTranscriptionErrorBody {
			errorBody = errorBody[:maxTranscriptionErrorBody]
		}
		message := strings.TrimSpace(string(errorBody))
		var apiError struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(errorBody, &apiError) == nil && apiError.Error.Message != "" {
			message = apiError.Error.Message
		}
		if message == "" {
			message = resp.Status
		}
		return nil, &transcriptionHTTPError{
			StatusCode: resp.StatusCode,
			Status:     resp.Status,
			Message:    message,
		}
	}
	return responseBody, nil
}
