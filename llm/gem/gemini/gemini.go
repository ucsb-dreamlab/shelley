package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"shelley.exe.dev/llm"
)

// https://ai.google.dev/api/generate-content#request-body
type Request struct {
	// Field order matters for JSON serialization - stable fields should come first
	// to maximize prefix deduplication when storing LLM requests.
	CachedContent     string            `json:"cachedContent,omitempty"` // format: "cachedContents/{name}"
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	Tools             []Tool            `json:"tools,omitempty"`
	// ToolConfig has been left out because it does not appear to be useful.
	// Contents comes last since it grows with each request in a conversation
	Contents []Content `json:"contents"`
}

// https://ai.google.dev/api/generate-content#response-body
type Response struct {
	Candidates    []Candidate    `json:"candidates"`
	UsageMetadata *UsageMetadata `json:"usageMetadata,omitempty"`
	headers       http.Header    // captured HTTP response headers
}

type UsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount    int `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount         int `json:"totalTokenCount,omitempty"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
}

// Header returns the HTTP response headers.
func (r *Response) Header() http.Header {
	return r.headers
}

type Candidate struct {
	Content      Content `json:"content"`
	FinishReason string  `json:"finishReason,omitempty"`
}

type Content struct {
	Parts []Part `json:"parts"`
	Role  string `json:"role,omitempty"`
}

// Part is a part of the content.
// This is a union data structure, only one-of the fields can be set.
type Part struct {
	Text                string               `json:"text,omitempty"`
	FunctionCall        *FunctionCall        `json:"functionCall,omitempty"`
	FunctionResponse    *FunctionResponse    `json:"functionResponse,omitempty"`
	ExecutableCode      *ExecutableCode      `json:"executableCode,omitempty"`
	CodeExecutionResult *CodeExecutionResult `json:"codeExecutionResult,omitempty"`
	// ThoughtSignature is required for Gemini 3 models when using function calling.
	// It must be passed back exactly as received when sending the conversation history.
	// Note: presence of ThoughtSignature does NOT mean the part is a thought summary —
	// Gemini 3 attaches it to ordinary final-answer text and tool calls so that
	// internal reasoning state can be rehydrated on the next turn. Use Thought to
	// detect a thought summary.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	// Thought is true when the part is a thought summary (only emitted when
	// thinkingConfig.includeThoughts is true). https://ai.google.dev/gemini-api/docs/thinking
	Thought    bool  `json:"thought,omitempty"`
	InlineData *Blob `json:"inlineData,omitempty"`
	// TODO fileData
}

// Blob is inline binary data (e.g. an image).
// https://ai.google.dev/api/caching#Blob
type Blob struct {
	MimeType string `json:"mimeType"`
	// Data is base64-encoded binary content.
	Data string `json:"data"`
}

type FunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type FunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type ExecutableCode struct {
	Language Language `json:"language"`
	Code     string   `json:"code"`
}

type Language int

const (
	LanguageUnspecified Language = 0
	LanguagePython      Language = 1 // python >= 3.10 with numpy and simpy
)

type CodeExecutionResult struct {
	Outcome Outcome `json:"outcome"`
	Output  string  `json:"output"`
}

type Outcome int

const (
	OutcomeUnspecified      Outcome = 0
	OutcomeOK               Outcome = 1
	OutcomeFailed           Outcome = 2
	OutcomeDeadlineExceeded Outcome = 3
)

// https://ai.google.dev/api/generate-content#v1beta.GenerationConfig
type GenerationConfig struct {
	ResponseMimeType string          `json:"responseMimeType,omitempty"` // text/plain, application/json, or text/x.enum
	ResponseSchema   *Schema         `json:"responseSchema,omitempty"`   // for JSON
	MaxOutputTokens  int             `json:"maxOutputTokens,omitempty"`
	ThinkingConfig   *ThinkingConfig `json:"thinkingConfig,omitempty"`
}

// ThinkingConfig controls extended thinking for Gemini models.
// ThinkingLevel and ThinkingBudget are mutually exclusive: setting both
// returns a 400 from the API. Use ThinkingLevel for Gemini 3.x and
// ThinkingBudget for Gemini 2.5.
// https://ai.google.dev/gemini-api/docs/thinking
type ThinkingConfig struct {
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`   // Gemini 3.x: "minimal", "low", "medium", "high"
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`  // Gemini 2.5: token count, -1 dynamic, 0 disable
	IncludeThoughts bool   `json:"includeThoughts,omitempty"` // include thought summaries in response
}

// https://ai.google.dev/api/caching#Tool
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations"`
	CodeExecution        *struct{}             `json:"codeExecution,omitempty"` // if present, enables the model to execute code
	// TODO googleSearchRetrieval https://ai.google.dev/api/caching#GoogleSearchRetrieval
}

// https://ai.google.dev/api/caching#FunctionDeclaration
type FunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  Schema `json:"parameters"`
}

// https://ai.google.dev/api/caching#Schema
type Schema struct {
	Type        DataType          `json:"type"`
	Format      string            `json:"string,omitempty"` // for NUMBER type: float, double for INTEGER type: int32, int64 for STRING type: enum
	Description string            `json:"description,omitempty"`
	Nullable    *bool             `json:"nullable,omitempty"`
	Enum        []string          `json:"enum,omitempty"`
	MaxItems    string            `json:"maxItems,omitempty"`   // for ARRAY
	MinItems    string            `json:"minItems,omitempty"`   // for ARRAY
	Properties  map[string]Schema `json:"properties,omitempty"` // for OBJECT
	Required    []string          `json:"required,omitempty"`   // for OBJECT
	Items       *Schema           `json:"items,omitempty"`      // for ARRAY
}

type DataType int

const (
	DataTypeUNSPECIFIED = DataType(0) // Not specified, should not be used.
	DataTypeSTRING      = DataType(1)
	DataTypeNUMBER      = DataType(2)
	DataTypeINTEGER     = DataType(3)
	DataTypeBOOLEAN     = DataType(4)
	DataTypeARRAY       = DataType(5)
	DataTypeOBJECT      = DataType(6)
)

// DefaultEndpoint is the Gemini API base URL used when Model.Endpoint is empty.
const DefaultEndpoint = "https://generativelanguage.googleapis.com/v1beta"

// APIError is returned by GenerateContent when the server returns a non-200 status.
type APIError struct {
	StatusCode int
	Body       string
	Header     http.Header
}

func (e *APIError) Error() string {
	return fmt.Sprintf("GenerateContent: HTTP status: %d, %s", e.StatusCode, e.Body)
}

type Model struct {
	Model    string // e.g. "models/gemini-1.5-flash"
	APIKey   string
	HTTPC    *http.Client          // if nil, http.DefaultClient is used
	Endpoint string                // if empty, DefaultEndpoint is used
	OnStream func(llm.StreamDelta) // optional callback for streamed text/thinking deltas
}

func (m Model) GenerateContent(ctx context.Context, req *Request) (*Response, error) {
	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/%s:streamGenerateContent?alt=sse", m.endpoint(), m.Model), bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("creating HTTP request: %w", err)
	}
	httpReq.Header.Add("Content-Type", "application/json")
	httpReq.Header.Add("x-goog-api-key", m.APIKey)
	httpResp, err := m.httpc().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("GenerateContent: do: %w", err)
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return nil, fmt.Errorf("GenerateContent: reading response body: %w", err)
		}
		return nil, &APIError{StatusCode: httpResp.StatusCode, Body: string(body), Header: httpResp.Header}
	}
	res, err := parseStreamResponse(ctx, httpResp.Body, m.OnStream)
	if err != nil {
		return nil, err
	}
	res.headers = httpResp.Header
	return res, nil
}

func parseStreamResponse(ctx context.Context, r io.Reader, onStream func(llm.StreamDelta)) (*Response, error) {
	scanner := bufio.NewScanner(r)
	// SSE lines can be large (e.g. tool call JSON or long thought/text chunks).
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	var (
		merged        Response
		dataLines     []string
		fallbackLines []string
		sawSSE        bool
	)

	processData := func(raw string) error {
		raw = strings.TrimSpace(raw)
		if raw == "" || raw == "[DONE]" {
			return nil
		}
		var chunk Response
		if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
			return fmt.Errorf("GenerateContent: unmarshaling SSE chunk: %w, %s", err, raw)
		}
		llm.ReportProgress(ctx)
		mergeResponseChunk(&merged, &chunk, onStream)
		return nil
	}

	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if len(dataLines) > 0 {
				raw := strings.Join(dataLines, "\n")
				dataLines = dataLines[:0]
				if err := processData(raw); err != nil {
					return nil, err
				}
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			// SSE comment / keepalive
			sawSSE = true
			fallbackLines = nil
			continue
		}
		if value, ok := strings.CutPrefix(line, "data:"); ok {
			sawSSE = true
			fallbackLines = nil
			value = strings.TrimPrefix(value, " ")
			dataLines = append(dataLines, value)
			joined := strings.TrimSpace(strings.Join(dataLines, "\n"))
			if joined == "[DONE]" {
				dataLines = dataLines[:0]
			} else if json.Valid([]byte(joined)) {
				dataLines = dataLines[:0]
				if err := processData(joined); err != nil {
					return nil, err
				}
			}
			continue
		}
		if strings.HasPrefix(line, "event:") || strings.HasPrefix(line, "id:") || strings.HasPrefix(line, "retry:") {
			sawSSE = true
			fallbackLines = nil
			continue
		}
		if !sawSSE {
			fallbackLines = append(fallbackLines, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("GenerateContent: reading response body: %w", err)
	}

	if len(dataLines) > 0 {
		raw := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		if err := processData(raw); err != nil {
			return nil, err
		}
	}

	if !sawSSE {
		body := strings.TrimSpace(strings.Join(fallbackLines, "\n"))
		if body == "" {
			return &merged, nil
		}
		var chunk Response
		if err := json.Unmarshal([]byte(body), &chunk); err != nil {
			return nil, fmt.Errorf("GenerateContent: unmarshaling response: %w, %s", err, body)
		}
		llm.ReportProgress(ctx)
		merged = chunk
		if onStream != nil && len(merged.Candidates) > 0 {
			for idx, part := range merged.Candidates[0].Content.Parts {
				if isTextOrSignatureOnly(part) && part.Text != "" {
					deltaType := "text"
					if part.Thought {
						deltaType = "thinking"
					}
					onStream(llm.StreamDelta{Type: deltaType, Text: part.Text, Index: idx})
				}
			}
		}
	}

	return &merged, nil
}

func isTextOrSignatureOnly(p Part) bool {
	return p.FunctionCall == nil &&
		p.FunctionResponse == nil &&
		p.ExecutableCode == nil &&
		p.CodeExecutionResult == nil &&
		p.InlineData == nil
}

func mergeResponseChunk(merged *Response, chunk *Response, onStream func(llm.StreamDelta)) {
	if chunk.UsageMetadata != nil {
		merged.UsageMetadata = chunk.UsageMetadata
	}
	for i, cand := range chunk.Candidates {
		for len(merged.Candidates) <= i {
			merged.Candidates = append(merged.Candidates, Candidate{})
		}
		dst := &merged.Candidates[i]
		if cand.Content.Role != "" {
			dst.Content.Role = cand.Content.Role
		}
		if cand.FinishReason != "" {
			dst.FinishReason = cand.FinishReason
		}
		for _, part := range cand.Content.Parts {
			dst.Content.Parts = mergePart(dst.Content.Parts, part, i == 0, onStream)
		}
	}
}

func mergePart(parts []Part, part Part, isPrimaryCandidate bool, onStream func(llm.StreamDelta)) []Part {
	if isTextOrSignatureOnly(part) {
		if part.Text == "" && part.ThoughtSignature == "" {
			return parts
		}
		if part.Text == "" && part.ThoughtSignature != "" {
			if len(parts) > 0 {
				parts[len(parts)-1].ThoughtSignature = part.ThoughtSignature
				return parts
			}
			return append(parts, part)
		}
		if len(parts) > 0 {
			prev := &parts[len(parts)-1]
			if isTextOrSignatureOnly(*prev) && (prev.Text == "" || prev.Thought == part.Thought) {
				prev.Text += part.Text
				prev.Thought = part.Thought
				if part.ThoughtSignature != "" {
					prev.ThoughtSignature = part.ThoughtSignature
				}
				if isPrimaryCandidate && onStream != nil {
					deltaType := "text"
					if part.Thought {
						deltaType = "thinking"
					}
					onStream(llm.StreamDelta{Type: deltaType, Text: part.Text, Index: len(parts) - 1})
				}
				return parts
			}
		}
		parts = append(parts, part)
		if isPrimaryCandidate && onStream != nil {
			deltaType := "text"
			if part.Thought {
				deltaType = "thinking"
			}
			onStream(llm.StreamDelta{Type: deltaType, Text: part.Text, Index: len(parts) - 1})
		}
		return parts
	}

	if len(parts) > 0 {
		prev := &parts[len(parts)-1]
		if isTextOrSignatureOnly(*prev) && prev.Text == "" && prev.ThoughtSignature != "" {
			if part.ThoughtSignature == "" {
				part.ThoughtSignature = prev.ThoughtSignature
			}
			*prev = part
			return parts
		}
	}
	return append(parts, part)
}

func (m Model) endpoint() string {
	if m.Endpoint != "" {
		return m.Endpoint
	}
	return DefaultEndpoint
}

func (m Model) httpc() *http.Client {
	if m.HTTPC != nil {
		return m.HTTPC
	}
	return http.DefaultClient
}
