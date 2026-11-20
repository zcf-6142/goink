package openairesponses

import (
	"context"
	"mime"
	"net/http"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"

	"github.com/sigpanic/goink/internal/llm/adapter"
)

// Stream 实时发布显示事件；完整终态与传输均成功后才交付状态和整组可执行工具。
func (c *Adapter) Stream(ctx context.Context, req adapter.Request) <-chan adapter.StreamEvent {
	ch := make(chan adapter.StreamEvent, 32)
	go func() {
		defer close(ch)
		emit := func(event adapter.StreamEvent) bool { return adapter.SendEvent(ctx, ch, event) }
		fail := func(err error) { emit(adapter.StreamEvent{Type: adapter.EventError, Error: err}) }
		p, err := c.provider(req.ProviderName)
		if err != nil {
			fail(err)
			return
		}
		params, err := p.buildRequest(req)
		if err != nil {
			fail(err)
			return
		}
		result, err := p.streamResponse(ctx, params, emit)
		if err != nil {
			fail(err)
			return
		}
		if !emit(adapter.StreamEvent{Type: adapter.EventUsage, Usage: result.Usage}) ||
			!emit(adapter.StreamEvent{Type: adapter.EventResponseState, State: result.State}) {
			return
		}
		for _, call := range result.ToolCalls {
			if !emit(adapter.StreamEvent{Type: adapter.EventToolCallEnd, Delta: &call}) {
				return
			}
		}
	}()
	return ch
}

func (p Provider) streamResponse(ctx context.Context, params responses.ResponseNewParams, emit func(adapter.StreamEvent) bool) (adapter.Response, error) {
	opts, err := responsesURLOptions(p.ChatURL)
	if err != nil {
		return adapter.Response{}, err
	}
	var response *http.Response
	opts = append(opts, option.WithHeader("Accept", "text/event-stream"), option.WithResponseInto(&response))
	stream := p.client.NewStreaming(ctx, params, opts...)
	defer stream.Close()
	if err := stream.Err(); err != nil {
		return adapter.Response{}, convertSDKError(err, response)
	}
	if response == nil {
		return adapter.Response{}, invalidResponse("missing SSE HTTP response")
	}
	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil || mediaType != "text/event-stream" {
		return adapter.Response{}, invalidResponse("expected text/event-stream response")
	}
	var state streamState
	for stream.Next() {
		if err := state.consume(stream.Current(), emit); err != nil {
			return adapter.Response{}, err
		}
		if err := ctx.Err(); err != nil {
			return adapter.Response{}, err
		}
	}
	if err := stream.Err(); err != nil {
		return adapter.Response{}, convertSDKError(err, response)
	}
	if err := ctx.Err(); err != nil {
		return adapter.Response{}, err
	}
	return state.finish(p, params)
}
