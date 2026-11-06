package gemini

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	"shelley.exe.dev/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestGenerateContentKeepsAPIKeyOutOfURL(t *testing.T) {
	const apiKey = "secret-api-key"
	var gotReq *http.Request
	m := Model{
		Model:  "models/gemini-test",
		APIKey: apiKey,
		HTTPC: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			gotReq = req
			return nil, errors.New("transport failed")
		})},
	}

	_, err := m.GenerateContent(t.Context(), &Request{})
	if err == nil {
		t.Fatal("GenerateContent succeeded, want transport error")
	}
	if gotReq == nil {
		t.Fatal("transport did not receive a request")
	}
	if got := gotReq.Header.Get("x-goog-api-key"); got != apiKey {
		t.Errorf("x-goog-api-key = %q, want API key", got)
	}
	if !strings.HasSuffix(gotReq.URL.String(), "/models/gemini-test:streamGenerateContent?alt=sse") {
		t.Errorf("request URL = %s, want suffix /models/gemini-test:streamGenerateContent?alt=sse", gotReq.URL)
	}
	if strings.Contains(gotReq.URL.String(), apiKey) {
		t.Errorf("request URL contains API key: %s", gotReq.URL)
	}
	if strings.Contains(err.Error(), apiKey) {
		t.Errorf("transport error contains API key: %v", err)
	}
}

func TestGenerateContentSSEStreaming(t *testing.T) {
	var gotPath, gotQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Exedev-Gateway-Cost", "0.0025")
		fmt.Fprint(w, ": keepalive\n\n")
		fmt.Fprint(w, `data: {"candidates": [{"content": {"role": "model","parts": [{"text": "1, 2"}]}}],"usageMetadata": {"promptTokenCount": 10}}`+"\n\n")
		fmt.Fprint(w, ": keepalive\n\n")
		fmt.Fprint(w, `data: {"candidates": [{"content": {"role": "model","parts": [{"text": ", 3, 4, 5."}]}}]}`+"\n\n")
		fmt.Fprint(w, `data: {"candidates": [{"content": {"role": "model","parts": [{"text": "","thoughtSignature": "AY89a19test"}]},"finishReason": "STOP"}],"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18}}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	var deltas []llm.StreamDelta
	m := Model{
		Model:    "models/gemini-4-argon",
		APIKey:   "test-key",
		Endpoint: server.URL,
		OnStream: func(d llm.StreamDelta) {
			deltas = append(deltas, d)
		},
	}

	progress := 0
	ctx := llm.WithProgress(t.Context(), func() { progress++ })
	res, err := m.GenerateContent(ctx, &Request{
		Contents: []Content{{Role: "user", Parts: []Part{{Text: "Count to 5"}}}},
	})
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if gotPath != "/models/gemini-4-argon:streamGenerateContent" || gotQuery != "alt=sse" {
		t.Errorf("request path/query = %q?%q, want /models/gemini-4-argon:streamGenerateContent?alt=sse", gotPath, gotQuery)
	}
	if progress != 3 {
		t.Errorf("progress reports = %d, want 3 (one per SSE data chunk)", progress)
	}
	if got := res.Header().Get("Exedev-Gateway-Cost"); got != "0.0025" {
		t.Errorf("Exedev-Gateway-Cost header = %q, want 0.0025", got)
	}
	if len(res.Candidates) != 1 {
		t.Fatalf("len(Candidates) = %d, want 1", len(res.Candidates))
	}
	cand := res.Candidates[0]
	if cand.FinishReason != "STOP" {
		t.Errorf("FinishReason = %q, want STOP", cand.FinishReason)
	}
	if cand.Content.Role != "model" {
		t.Errorf("Role = %q, want model", cand.Content.Role)
	}
	wantParts := []Part{{
		Text:             "1, 2, 3, 4, 5.",
		ThoughtSignature: "AY89a19test",
	}}
	if !reflect.DeepEqual(cand.Content.Parts, wantParts) {
		t.Errorf("Parts = %+v, want %+v", cand.Content.Parts, wantParts)
	}
	wantDeltas := []llm.StreamDelta{
		{Type: "text", Text: "1, 2", Index: 0},
		{Type: "text", Text: ", 3, 4, 5.", Index: 0},
	}
	if !reflect.DeepEqual(deltas, wantDeltas) {
		t.Errorf("deltas = %+v, want %+v", deltas, wantDeltas)
	}
	if res.UsageMetadata == nil || res.UsageMetadata.TotalTokenCount != 18 {
		t.Errorf("UsageMetadata = %+v, want TotalTokenCount=18", res.UsageMetadata)
	}
}

func TestGenerateContentSSEThinkingTextAndToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range []string{
			`{"candidates": [{"content": {"role": "model","parts": [{"thought": true, "text": "Thinking step 1. "}]}}]}`,
			`{"candidates": [{"content": {"role": "model","parts": [{"thought": true, "text": "Thinking step 2."}]}}]}`,
			`{"candidates": [{"content": {"role": "model","parts": [{"thought": true, "text": "", "thoughtSignature": "sig-thought"}]}}]}`,
			`{"candidates": [{"content": {"role": "model","parts": [{"text": "Let me run a tool"}]}}]}`,
			`{"candidates": [{"content": {"role": "model","parts": [{"text": " now."}]}}]}`,
			`{"candidates": [{"content": {"role": "model","parts": [{"functionCall": {"name": "bash", "args": {"command": "ls"}}}]} }]}`,
			`{"candidates": [{"content": {"role": "model","parts": [{"text": "", "thoughtSignature": "sig-fc"}]}, "finishReason": "STOP"}]}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}
	}))
	defer server.Close()

	var deltas []llm.StreamDelta
	m := Model{
		Model:    "models/gemini-4-argon",
		APIKey:   "test-key",
		Endpoint: server.URL,
		OnStream: func(d llm.StreamDelta) {
			deltas = append(deltas, d)
		},
	}

	progress := 0
	ctx := llm.WithProgress(t.Context(), func() { progress++ })
	res, err := m.GenerateContent(ctx, &Request{})
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if progress != 7 {
		t.Errorf("progress = %d, want 7", progress)
	}
	if len(res.Candidates) != 1 {
		t.Fatalf("len(Candidates) = %d, want 1", len(res.Candidates))
	}
	parts := res.Candidates[0].Content.Parts
	if len(parts) != 3 {
		t.Fatalf("len(Parts) = %d, want 3: %+v", len(parts), parts)
	}
	if !parts[0].Thought || parts[0].Text != "Thinking step 1. Thinking step 2." || parts[0].ThoughtSignature != "sig-thought" {
		t.Errorf("parts[0] = %+v, want merged thought with sig-thought", parts[0])
	}
	if parts[1].Thought || parts[1].Text != "Let me run a tool now." {
		t.Errorf("parts[1] = %+v, want merged text", parts[1])
	}
	if parts[2].FunctionCall == nil || parts[2].FunctionCall.Name != "bash" || parts[2].ThoughtSignature != "sig-fc" {
		t.Errorf("parts[2] = %+v, want FunctionCall bash with sig-fc", parts[2])
	}
	wantDeltas := []llm.StreamDelta{
		{Type: "thinking", Text: "Thinking step 1. ", Index: 0},
		{Type: "thinking", Text: "Thinking step 2.", Index: 0},
		{Type: "text", Text: "Let me run a tool", Index: 1},
		{Type: "text", Text: " now.", Index: 1},
	}
	if !reflect.DeepEqual(deltas, wantDeltas) {
		t.Errorf("deltas = %+v, want %+v", deltas, wantDeltas)
	}
}

func TestGenerateContent(t *testing.T) {
	// TODO replace with local replay endpoint
	m := Model{
		Model:  "models/gemini-3.6-flash",
		APIKey: os.Getenv("GEMINI_API_KEY"),
	}
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	if m.APIKey == "" {
		t.Skip("skipping test without API key")
	}

	res, err := m.GenerateContent(t.Context(), &Request{
		Contents: []Content{{
			Parts: []Part{{
				Text: "What is the capital of France?",
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("res: %+v", res)
}
