package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"shelley.exe.dev/models"
	"shelley.exe.dev/modelsources"
)

func testOpenAIRecordingTranscriber(client *http.Client, endpoint string) *openAIRecordingTranscriber {
	return &openAIRecordingTranscriber{
		client: client,
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			return models.TranscriptionModel{Model: modelName, Endpoint: endpoint}, nil
		},
	}
}

func TestOpenAIRecordingTranscriber(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "direct.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("authorization should be injected by the proxy, got %q", got)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if got := r.FormValue("model"); got != "gpt-transcribe" {
			t.Errorf("model = %q", got)
		}
		if got := r.FormValue("response_format"); got != "json" {
			t.Errorf("response_format = %q", got)
		}
		if got := r.MultipartForm.Value["timestamp_granularities[]"]; len(got) != 0 {
			t.Errorf("timestamp granularities = %#v", got)
		}
		if got := r.FormValue("prompt"); got != "Shelley on example-vm" {
			t.Errorf("prompt = %q", got)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		if header.Filename != "direct.webm" {
			t.Errorf("filename = %q", header.Filename)
		}
		data, err := io.ReadAll(file)
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != "media" {
			t.Errorf("file = %q", data)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":" immediate words "}`)
	}))
	defer api.Close()

	transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
	result, err := transcriber.Transcribe(t.Context(), mediaPath, "Shelley on example-vm", false)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "immediate words" || result.Model != "gpt-transcribe" {
		t.Fatalf("result = %#v", result)
	}
	if result.TimestampsModel != "" || result.TimestampsPath != "" {
		t.Fatalf("unexpected timestamps result = %#v", result)
	}
}

func TestOpenAIRecordingTranscriberUsesModelAPIKey(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "authenticated.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("authorization = %q", got)
		}
		_, _ = io.WriteString(w, `{"text":"authenticated words"}`)
	}))
	defer api.Close()

	transcriber := &openAIRecordingTranscriber{
		client: api.Client(),
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			return models.TranscriptionModel{Model: modelName, Endpoint: api.URL, APIKey: "secret"}, nil
		},
	}
	if _, err := transcriber.Transcribe(t.Context(), mediaPath, "context", false); err != nil {
		t.Fatal(err)
	}
}

// An OpenAI credential (here via the gateway source with an explicit key)
// yields transcription routes, just as it yields chat models.
func TestOpenAIRecordingTranscriberUsesOpenAICredentialRoutes(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "keyed.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/openai/v1/audio/transcriptions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("authorization = %q", got)
		}
		_, _ = io.WriteString(w, `{"text":"keyed words"}`)
	}))
	defer api.Close()

	manager, err := models.NewManager(&models.Config{
		TranscriptionModels: modelsources.TranscriptionModels([]modelsources.Source{modelsources.Gateway(api.URL, "", "sk-test", "")}),
	})
	if err != nil {
		t.Fatal(err)
	}
	transcriber := newOpenAIRecordingTranscriber(manager, false)
	if err := transcriber.Available(); err != nil {
		t.Fatal(err)
	}
	result, err := transcriber.Transcribe(t.Context(), mediaPath, "context", false)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "keyed words" {
		t.Fatalf("result = %#v", result)
	}
}

func TestOpenAIRecordingTranscriberUnavailableWithoutRoutes(t *testing.T) {
	manager, err := models.NewManager(&models.Config{})
	if err != nil {
		t.Fatal(err)
	}
	err = newOpenAIRecordingTranscriber(manager, false).Available()
	if err == nil || !strings.Contains(err.Error(), `"gpt-transcribe" is not available`) || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Fatalf("error = %v", err)
	}
}

// Screen and review recordings need whisper-1 too; accepting them with only
// gpt-transcribe would queue an item that can only fail.
func TestOpenAIRecordingTranscriberUnavailableWithoutTimestampsModel(t *testing.T) {
	manager, err := models.NewManager(&models.Config{TranscriptionModels: []models.TranscriptionModel{{
		Model:    models.TranscriptionTextModel,
		Endpoint: "https://api.openai.com/v1/audio/transcriptions",
	}}})
	if err != nil {
		t.Fatal(err)
	}
	err = newOpenAIRecordingTranscriber(manager, false).Available()
	if !errors.Is(err, errNoTranscriptionRoute) || !strings.Contains(err.Error(), `"whisper-1"`) {
		t.Fatalf("error = %v", err)
	}
}

func TestOpenAIRecordingTranscriberIgnoresRoutesWhenPredictableOnly(t *testing.T) {
	manager, err := models.NewManager(&models.Config{
		TranscriptionModels: modelsources.TranscriptionModels([]modelsources.Source{modelsources.Env("", "sk-test", "", "")}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := newOpenAIRecordingTranscriber(manager, false).Available(); err != nil {
		t.Fatal(err)
	}
	if err := newOpenAIRecordingTranscriber(manager, true).Available(); !errors.Is(err, errNoTranscriptionRoute) {
		t.Fatalf("error = %v", err)
	}
}

func TestOpenAIRecordingTranscriberWithTimestamps(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "screen.webm")
	var gptRequests, whisperRequests atomic.Int32
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		switch model := r.FormValue("model"); model {
		case "gpt-transcribe":
			gptRequests.Add(1)
			if got := r.FormValue("response_format"); got != "json" {
				t.Errorf("GPT response_format = %q", got)
			}
			if got := r.MultipartForm.Value["timestamp_granularities[]"]; len(got) != 0 {
				t.Errorf("GPT timestamp granularities = %#v", got)
			}
			_, _ = io.WriteString(w, `{"text":" adjusted GPT words "}`)
		case models.TranscriptionTimestampsModel:
			whisperRequests.Add(1)
			if got := r.FormValue("response_format"); got != "verbose_json" {
				t.Errorf("Whisper response_format = %q", got)
			}
			if got := r.MultipartForm.Value["timestamp_granularities[]"]; len(got) != 2 || got[0] != "word" || got[1] != "segment" {
				t.Errorf("Whisper timestamp granularities = %#v", got)
			}
			_, _ = io.WriteString(w, `{"text":" literal whisper words ","words":[{"word":"literal","start":0.0,"end":0.4}],"segments":[{"text":"literal whisper words","start":0.0,"end":0.8}]}`)
		default:
			t.Errorf("unexpected model = %q", model)
		}
		w.Header().Set("Content-Type", "application/json")
	}))
	defer api.Close()

	transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
	result, err := transcriber.Transcribe(t.Context(), mediaPath, "Shelley on example-vm", true)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "adjusted GPT words" ||
		result.Model != "gpt-transcribe" ||
		result.TimestampsModel != models.TranscriptionTimestampsModel {
		t.Fatalf("result = %#v", result)
	}
	if gptRequests.Load() != 1 || whisperRequests.Load() != 1 {
		t.Fatalf("requests: GPT=%d Whisper=%d", gptRequests.Load(), whisperRequests.Load())
	}
	wantTimestampsPath := mediaPath + ".timestamps.json"
	if result.TimestampsPath != wantTimestampsPath {
		t.Fatalf("timestamps path = %q, want %q", result.TimestampsPath, wantTimestampsPath)
	}
	timestamps, err := os.ReadFile(wantTimestampsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(timestamps), `"literal whisper words"`) ||
		!strings.Contains(string(timestamps), `"words"`) ||
		!strings.Contains(string(timestamps), `"segments"`) {
		t.Fatalf("timestamps = %s", timestamps)
	}
}

func TestOpenAIRecordingTranscriberReportsAPIError(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "error.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"unsupported recording"}}`)
	}))
	defer api.Close()

	transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
	_, err := transcriber.Transcribe(context.Background(), mediaPath, "context", false)
	if err == nil || !strings.Contains(err.Error(), "unsupported recording") {
		t.Fatalf("error = %v", err)
	}
}

func TestSelectTranscriptionModelPrefersLLMIntegration(t *testing.T) {
	selected, err := selectTranscriptionModel(models.TranscriptionTextModel, []models.TranscriptionModel{
		{Model: models.TranscriptionTextModel, Endpoint: "https://custom.example/v1/audio/transcriptions", Source: models.SourceCustomLabel},
		{Model: models.TranscriptionTextModel, Endpoint: "https://ai.team.exe.xyz/v1/audio/transcriptions", Source: "ai.team.exe.xyz"},
		{Model: models.TranscriptionTextModel, Endpoint: "https://llm.int.exe.xyz/v1/audio/transcriptions", Source: "llm.int.exe.xyz"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Endpoint != "https://llm.int.exe.xyz/v1/audio/transcriptions" {
		t.Fatalf("endpoint = %q", selected.Endpoint)
	}
}

// Any integration, not just the one named llm, wins over a custom model.
func TestSelectTranscriptionModelPrefersTeamIntegrationOverCustom(t *testing.T) {
	selected, err := selectTranscriptionModel(models.TranscriptionTextModel, []models.TranscriptionModel{
		{Model: models.TranscriptionTextModel, Endpoint: "https://custom.example/v1/audio/transcriptions", Source: models.SourceCustomLabel},
		{Model: models.TranscriptionTextModel, Endpoint: "https://ai.team.exe.xyz/v1/audio/transcriptions", Source: "ai.team.exe.xyz"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Endpoint != "https://ai.team.exe.xyz/v1/audio/transcriptions" {
		t.Fatalf("endpoint = %q", selected.Endpoint)
	}
}

// A custom model is the user's explicit choice; it wins over routes derived
// from OPENAI_API_KEY or the gateway.
func TestSelectTranscriptionModelPrefersCustomOverCredentialRoute(t *testing.T) {
	selected, err := selectTranscriptionModel(models.TranscriptionTextModel, []models.TranscriptionModel{
		{Model: models.TranscriptionTextModel, Endpoint: "https://api.openai.com/v1/audio/transcriptions", Source: "$OPENAI_API_KEY", FromCredentials: true},
		{Model: models.TranscriptionTextModel, Endpoint: "https://proxy.example/v1/audio/transcriptions", Source: models.SourceCustomLabel},
	})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Endpoint != "https://proxy.example/v1/audio/transcriptions" {
		t.Fatalf("endpoint = %q", selected.Endpoint)
	}
}

func TestSelectTranscriptionModelRequiresExactKnownModel(t *testing.T) {
	_, err := selectTranscriptionModel(models.TranscriptionTextModel, []models.TranscriptionModel{
		{Model: "some-other-model", Endpoint: "https://llm.int.exe.xyz/v1/audio/transcriptions"},
	})
	if err == nil || !strings.Contains(err.Error(), "not available") {
		t.Fatalf("error = %v", err)
	}
}

func TestOpenAIRecordingTranscriberDoesNotRetryOrdinaryErrors(t *testing.T) {
	for name, response := range map[string]struct {
		status  int
		body    string
		wantErr string
	}{
		"credits":     {http.StatusPaymentRequired, `{"error":{"message":"credits exhausted"}}`, "credits exhausted"},
		"bad_request": {http.StatusBadRequest, `{"error":{"message":"unsupported recording"}}`, "unsupported recording"},
		"forbidden":   {http.StatusForbidden, `{"error":{"message":"access forbidden"}}`, "access forbidden"},
		"empty":       {http.StatusOK, `{"text":""}`, "empty transcript"},
	} {
		t.Run(name, func(t *testing.T) {
			mediaPath := transcriptionTestFile(t, name+".webm")
			first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(response.status)
				_, _ = io.WriteString(w, response.body)
			}))
			defer first.Close()
			var secondCalls atomic.Int32
			second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				secondCalls.Add(1)
				_, _ = io.WriteString(w, `{"text":"must not run"}`)
			}))
			defer second.Close()

			transcriber := testOpenAIRecordingTranscriber(first.Client(), first.URL)
			_, err := transcriber.Transcribe(t.Context(), mediaPath, "context", false)
			if err == nil || !strings.Contains(err.Error(), response.wantErr) {
				t.Fatalf("error = %v", err)
			}
			if secondCalls.Load() != 0 {
				t.Fatalf("second gateway requests = %d, want 0", secondCalls.Load())
			}
		})
	}
}

func TestOpenAIRecordingTranscriberDoesNotRetryCancelledRequest(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "cancelled-request.webm")
	var requests atomic.Int32
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		_, _ = io.WriteString(w, `{"text":"must not run"}`)
	}))
	defer api.Close()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
	if _, err := transcriber.Transcribe(ctx, mediaPath, "context", false); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
	if requests.Load() != 0 {
		t.Fatalf("requests = %d, want 0", requests.Load())
	}
}

func TestOpenAIRecordingTranscriberRequiresTimestampArrays(t *testing.T) {
	for name, response := range map[string]string{
		"missing":    `{"text":"words without timings"}`,
		"null":       `{"text":"","words":null,"segments":[]}`,
		"wrong_type": `{"text":"","words":[],"segments":{}}`,
	} {
		t.Run(name, func(t *testing.T) {
			mediaPath := transcriptionTestFile(t, name+".webm")
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Fatal(err)
				}
				if r.FormValue("model") == models.TranscriptionTextModel {
					_, _ = io.WriteString(w, `{"text":"canonical GPT words"}`)
					return
				}
				_, _ = io.WriteString(w, response)
			}))
			defer api.Close()

			transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
			if _, err := transcriber.Transcribe(t.Context(), mediaPath, "context", true); err == nil ||
				(!strings.Contains(err.Error(), "must be an array") &&
					!strings.Contains(err.Error(), "cannot unmarshal")) {
				t.Fatalf("error = %v", err)
			}
			if _, err := os.Stat(mediaPath + ".timestamps.json"); !os.IsNotExist(err) {
				t.Fatalf("timestamp artifact should not exist: %v", err)
			}
		})
	}
}

func TestOpenAIRecordingTranscriberAcceptsEmptyTimingArraysForSilence(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "silence.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("model") == models.TranscriptionTextModel {
			_, _ = io.WriteString(w, `{"text":"canonical GPT words"}`)
			return
		}
		_, _ = io.WriteString(w, `{"text":"","words":[],"segments":[]}`)
	}))
	defer api.Close()

	transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
	if _, err := transcriber.Transcribe(t.Context(), mediaPath, "context", true); err != nil {
		t.Fatal(err)
	}
}

func TestOpenAIRecordingTranscriberAttributesTimestampFailure(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "timestamp-failure.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("model") == models.TranscriptionTextModel {
			_, _ = io.WriteString(w, `{"text":"canonical GPT words"}`)
			return
		}
		_, _ = io.WriteString(w, `{"text":""}`)
	}))
	defer api.Close()

	transcriber := testOpenAIRecordingTranscriber(api.Client(), api.URL)
	_, err := transcriber.Transcribe(t.Context(), mediaPath, "context", true)
	var modelError *transcriptionModelError
	if !errors.As(err, &modelError) || modelError.Model != models.TranscriptionTimestampsModel {
		t.Fatalf("error = %#v", err)
	}
}

func TestOpenAIRecordingTranscriberPreparesOversizedRecording(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "large.webm")
	if err := os.Truncate(mediaPath, maxTranscriptionUpload+1); err != nil {
		t.Fatal(err)
	}
	var uploaded atomic.Int64
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		_ = file.Close()
		uploaded.Store(header.Size)
		if filepath.Ext(header.Filename) != ".m4a" {
			t.Errorf("prepared filename = %q", header.Filename)
		}
		_, _ = io.WriteString(w, `{"text":"prepared words"}`)
	}))
	defer api.Close()

	transcriber := &openAIRecordingTranscriber{
		client: api.Client(),
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			return models.TranscriptionModel{Model: modelName, Endpoint: api.URL}, nil
		},
		mediaRun: func(_ context.Context, name string, args ...string) ([]byte, error) {
			if name != "ffmpeg" {
				return nil, fmt.Errorf("command = %q", name)
			}
			return nil, os.WriteFile(args[len(args)-1], []byte("compressed audio"), 0o600)
		},
	}
	result, err := transcriber.Transcribe(t.Context(), mediaPath, "context", false)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "prepared words" || uploaded.Load() != int64(len("compressed audio")) {
		t.Fatalf("result = %#v, uploaded = %d", result, uploaded.Load())
	}
}

func TestOpenAIRecordingTranscriberPreparesOversizedScreenRecordingOnce(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "large-screen.webm")
	if err := os.Truncate(mediaPath, maxTranscriptionUpload+1); err != nil {
		t.Fatal(err)
	}
	var preparationCalls atomic.Int32
	var preparedPath string
	var mu sync.Mutex
	uploadedFiles := make(map[string]string)
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(file)
		_ = file.Close()
		if err != nil {
			t.Fatal(err)
		}
		model := r.FormValue("model")
		mu.Lock()
		uploadedFiles[model] = header.Filename
		mu.Unlock()
		if string(data) != "compressed screen audio" {
			t.Errorf("%s upload = %q", model, data)
		}
		if model == models.TranscriptionTextModel {
			_, _ = io.WriteString(w, `{"text":"canonical screen words"}`)
			return
		}
		_, _ = io.WriteString(w, `{"text":"","words":[],"segments":[]}`)
	}))
	defer api.Close()

	transcriber := &openAIRecordingTranscriber{
		client: api.Client(),
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			return models.TranscriptionModel{Model: modelName, Endpoint: api.URL}, nil
		},
		mediaRun: func(_ context.Context, name string, args ...string) ([]byte, error) {
			if name != "ffmpeg" {
				return nil, fmt.Errorf("command = %q", name)
			}
			preparationCalls.Add(1)
			preparedPath = args[len(args)-1]
			return nil, os.WriteFile(preparedPath, []byte("compressed screen audio"), 0o600)
		},
	}
	result, err := transcriber.Transcribe(t.Context(), mediaPath, "context", true)
	if err != nil {
		t.Fatal(err)
	}
	if preparationCalls.Load() != 1 {
		t.Fatalf("preparation calls = %d, want 1", preparationCalls.Load())
	}
	if result.Text != "canonical screen words" || result.TimestampsPath != mediaPath+".timestamps.json" {
		t.Fatalf("result = %#v", result)
	}
	mu.Lock()
	gptFile := uploadedFiles[models.TranscriptionTextModel]
	whisperFile := uploadedFiles[models.TranscriptionTimestampsModel]
	mu.Unlock()
	if filepath.Ext(gptFile) != ".m4a" || gptFile != whisperFile {
		t.Fatalf("uploaded files: GPT=%q Whisper=%q", gptFile, whisperFile)
	}
	if _, err := os.Stat(result.TimestampsPath); err != nil {
		t.Fatalf("timestamps missing beside original media: %v", err)
	}
	if _, err := os.Stat(preparedPath); !os.IsNotExist(err) {
		t.Fatalf("prepared audio was not removed: %v", err)
	}
}

func TestOpenAIRecordingTranscriberRejectsStillOversizedPreparedAudio(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "still-large.webm")
	if err := os.Truncate(mediaPath, maxTranscriptionUpload+1); err != nil {
		t.Fatal(err)
	}
	var preparedPath string
	transcriber := &openAIRecordingTranscriber{
		client: http.DefaultClient,
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			return models.TranscriptionModel{Model: modelName, Endpoint: "http://unused.invalid"}, nil
		},
		mediaRun: func(_ context.Context, _ string, args ...string) ([]byte, error) {
			preparedPath = args[len(args)-1]
			return nil, os.Truncate(preparedPath, maxTranscriptionUpload+1)
		},
	}
	if _, err := transcriber.Transcribe(t.Context(), mediaPath, "context", false); err == nil ||
		!strings.Contains(err.Error(), "split it or use the transcribing-audio skill") {
		t.Fatalf("error = %v", err)
	}
	if _, err := os.Stat(preparedPath); !os.IsNotExist(err) {
		t.Fatalf("rejected prepared audio was not removed: %v", err)
	}
}

func TestOpenAIRecordingTranscriberGeminiAliasWithTimestamps(t *testing.T) {
	mediaPath := transcriptionTestFile(t, "gemini-screen.webm")
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer litellm-secret" {
			t.Errorf("authorization = %q", got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(string(body), `"json_object"`) {
			_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"{\"text\":\"gemini words\",\"words\":[{\"word\":\"gemini\",\"start\":0.1,\"end\":0.5}],\"segments\":[{\"start\":0.1,\"end\":0.9}]}"}}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"gemini words"}}]}`)
	}))
	defer api.Close()

	transcriber := &openAIRecordingTranscriber{
		client: api.Client(),
		resolveModel: func(modelName string) (models.TranscriptionModel, error) {
			return models.TranscriptionModel{
				Model:     modelName,
				WireModel: models.TranscriptionGeminiAlias,
				Endpoint:  api.URL + "/v1/chat/completions",
				APIKey:    "litellm-secret",
			}, nil
		},
	}
	result, err := transcriber.Transcribe(t.Context(), mediaPath, "Shelley on example-vm", true)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "gemini words" || result.Model != models.TranscriptionTextModel || result.TimestampsModel != models.TranscriptionTimestampsModel {
		t.Fatalf("result = %#v", result)
	}
	timestamps, err := os.ReadFile(result.TimestampsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(timestamps), `"words"`) || !strings.Contains(string(timestamps), `"segments"`) {
		t.Fatalf("timestamps = %s", timestamps)
	}
}

