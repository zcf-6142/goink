package openairesponses

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/sigpanic/goink/internal/llm/adapter"
)

func TestCodexCompatibilityIsScopedToProviderAcrossAllOperations(t *testing.T) {
	fixture := recordedStream(t)
	attempts := make(map[string]int)
	var attemptsMu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")[0]
		compat := name == "codex"
		if strings.Contains(r.UserAgent(), "Codex Responses compatible") != compat ||
			(compat && !strings.HasPrefix(r.UserAgent(), "Goink ")) {
			t.Error("compatibility identifier leaked across providers or lost Goink identity")
		}
		if r.Header.Get("Authorization") != "Bearer "+name+"-key" || r.URL.RawQuery != "route=a%2Fb&x=1&x=2" {
			t.Error("provider credentials or complete endpoint query changed")
		}
		attemptsMu.Lock()
		attempts[name]++
		attemptsMu.Unlock()
		if r.Method == http.MethodGet {
			if r.URL.Path != "/"+name+"/v1/models" {
				t.Error("model discovery used the wrong route")
			}
			w.Header().Set("Content-Type", "application/json")
			model := "gpt-5.6-sol"
			if compat {
				model = "gpt-6.1-sol"
			}
			fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"object":"model","created":1,"owned_by":"relay"}]}`, model)
			return
		}
		var body struct {
			Stream      bool
			Background  *bool
			Temperature *float64
			Input       json.RawMessage
			Reasoning   struct{ Effort string }
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if compat && !body.Stream {
			t.Error("compatibility generation sent a non-streaming request")
		}
		if compat && body.Background != nil || !compat && (body.Background == nil || *body.Background) {
			t.Error("background parameter omission was not scoped to compatibility mode")
		}
		if body.Temperature != nil {
			t.Error("reasoning request included an unsupported temperature")
		}
		if compat && len(body.Input) > 0 && body.Input[0] == '[' && body.Reasoning.Effort != "medium" {
			t.Error("omitting temperature changed the requested reasoning effort")
		}
		if body.Stream {
			if r.Header.Get("Accept") != "text/event-stream" {
				t.Error("streaming request lost its Accept header")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, fixture)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, generatedToolResponse)
	}))
	defer server.Close()
	backend, err := New(adapter.ProviderMap{
		"standard": {Name: "standard", ProviderType: "openai", ChatURL: server.URL + "/standard/v1/responses?route=a%2Fb&x=1&x=2", APIKey: "standard-key"},
		"codex":    {Name: "codex", ProviderType: ProviderTypeCodex, ChatURL: server.URL + "/codex/v1/responses?route=a%2Fb&x=1&x=2", APIKey: "codex-key", Temperature: pointer(0.8)},
	}, server.Client(), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"codex", "standard"} {
		req := streamRequest()
		req.ProviderName = name
		if name == "codex" {
			req.Model = "gpt-6.1-sol"
		}
		result, err := backend.Generate(context.Background(), req)
		if err != nil || result.State == nil || !result.State.Complete || len(result.ToolCalls) != 2 {
			t.Fatalf("%s generation did not preserve complete tool output: %+v %v", name, result, err)
		}
		if name == "codex" && (result.Content != "previewno" || result.ThinkingContent != "plan" || result.Usage["total_tokens"] != float64(33)) {
			t.Fatalf("compatibility generation lost final text, reasoning or usage: %+v", result)
		}
		for _, event := range collectStream(t, backend, req) {
			if event.Type == adapter.EventError {
				t.Fatal(event.Error)
			}
		}
		probe, err := backend.TestConnection(context.Background(), name, req.Model)
		if err != nil || probe.URL != backend.Providers()[name].ChatURL {
			t.Fatalf("%s connection probe failed: %+v %v", name, probe, err)
		}
		models, err := backend.DiscoverModels(context.Background(), name)
		if err != nil || len(models) != 1 || models[0].ID != req.Model {
			t.Fatalf("%s discovery failed: %+v %v", name, models, err)
		}
	}
	attemptsMu.Lock()
	defer attemptsMu.Unlock()
	for name, count := range attempts {
		if count != 4 {
			t.Errorf("%s: expected four single requests, got %d", name, count)
		}
	}
}

func TestCodexGenerationPreservesNativeToolReplay(t *testing.T) {
	var requests []string
	backend, err := New(adapter.ProviderMap{"relay": {
		Name: "relay", ProviderType: ProviderTypeCodex, ChatURL: "https://relay.invalid/v1/responses", APIKey: "test-key",
	}}, &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body struct{ Input json.RawMessage }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		requests = append(requests, string(body.Input))
		response := jsonResponse(recordedStream(t))
		response.Header.Set("Content-Type", "text/event-stream")
		return response, nil
	})}, nil)
	if err != nil {
		t.Fatal(err)
	}
	req := streamRequest()
	first, err := backend.Generate(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	message := adapter.Message{Role: "assistant", Content: first.Content, ThinkingContent: first.ThinkingContent, State: first.State}
	for _, call := range first.ToolCalls {
		message.ToolCalls = append(message.ToolCalls, adapter.ToolCall{ID: call.ToolID, Name: call.ToolName, Arguments: call.ArgumentsJSON})
	}
	req.Messages = append(req.Messages, message)
	for _, call := range first.ToolCalls {
		req.Messages = append(req.Messages, adapter.Message{Role: "tool", ToolCallID: call.ToolID, ToolName: call.ToolName, Content: "test result"})
	}
	if _, err := backend.Generate(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(requests) != 2 || !strings.Contains(requests[1], "complete-ciphertext") ||
		!strings.Contains(requests[1], `"call_id":"call_1"`) || !strings.Contains(requests[1], `"call_id":"call_2"`) ||
		!strings.Contains(requests[1], `"type":"function_call_output"`) || !strings.Contains(requests[1], "9007199254740993") {
		t.Fatal("synchronous streaming aggregation lost native reasoning or tool replay")
	}
}

func TestCodexGenerationRejectsTruncatedStreams(t *testing.T) {
	fixture := recordedStream(t)
	fixture = fixture[:strings.Index(fixture, "event: response.completed")]
	backend, err := New(adapter.ProviderMap{"relay": {
		Name: "relay", ProviderType: ProviderTypeCodex, ChatURL: "https://relay.invalid/v1/responses",
	}}, &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		response := jsonResponse(fixture)
		response.Header.Set("Content-Type", "text/event-stream")
		return response, nil
	})}, nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := backend.Generate(context.Background(), streamRequest())
	if err == nil || result.State != nil || len(result.ToolCalls) != 0 {
		t.Fatalf("truncated generation released partial executable tools: %+v %v", result, err)
	}
}
