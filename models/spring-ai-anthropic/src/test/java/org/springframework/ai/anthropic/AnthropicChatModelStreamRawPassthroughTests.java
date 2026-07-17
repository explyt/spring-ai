/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.anthropic;

import java.util.List;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.DualStreamItem;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.http.codec.ServerSentEvent;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * Unit tests for the Anthropic raw-response passthrough tee
 * ({@code AnthropicChatModel#streamRawPassthrough}): a single upstream SSE stream must be
 * emitted BOTH as verbatim {@link DualStreamItem.RawFrame}s - preserving the Anthropic
 * {@code event:} names, ping events, error events and unknown fields - AND as
 * {@link DualStreamItem.TypedChunk}s with correctly parsed usage.
 */
public class AnthropicChatModelStreamRawPassthroughTests {

	private static final String MODEL = "claude-sonnet-4-5";

	// message_start carries the input token count and a vendor-extension field the
	// typed DTO does not know about: it must survive verbatim in the raw branch.
	private static final String MESSAGE_START = "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\","
			+ "\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"" + MODEL + "\","
			+ "\"usage\":{\"input_tokens\":9,\"output_tokens\":1}},\"x_vendor_extension\":\"keep-me\"}";

	private static final String PING = "{\"type\":\"ping\"}";

	private static final String CONTENT_BLOCK_START = "{\"type\":\"content_block_start\",\"index\":0,"
			+ "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}";

	private static final String CONTENT_BLOCK_DELTA = "{\"type\":\"content_block_delta\",\"index\":0,"
			+ "\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello there\"}}";

	private static final String CONTENT_BLOCK_STOP = "{\"type\":\"content_block_stop\",\"index\":0}";

	private static final String MESSAGE_DELTA = "{\"type\":\"message_delta\","
			+ "\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}";

	private static final String MESSAGE_STOP = "{\"type\":\"message_stop\"}";

	private AnthropicChatModel chatModel;

	private AnthropicApi apiSpy;

	private void setupChatModel(Flux<ServerSentEvent<String>> sseFrames) {
		AnthropicApi realApi = AnthropicApi.builder().apiKey("test").build();
		this.apiSpy = spy(realApi);
		doReturn(sseFrames).when(this.apiSpy).chatCompletionStreamRawSse(anyString(), any());
		this.chatModel = AnthropicChatModel.builder()
			.anthropicApi(this.apiSpy)
			.defaultOptions(AnthropicChatOptions.builder().model(MODEL).maxTokens(4096).build())
			.build();
	}

	private static ServerSentEvent<String> sse(String event, String data) {
		return ServerSentEvent.builder(data).event(event).build();
	}

	private static Flux<ServerSentEvent<String>> defaultFrames() {
		return Flux.just(sse("message_start", MESSAGE_START), sse("ping", PING),
				sse("content_block_start", CONTENT_BLOCK_START), sse("content_block_delta", CONTENT_BLOCK_DELTA),
				sse("content_block_stop", CONTENT_BLOCK_STOP), sse("message_delta", MESSAGE_DELTA),
				sse("message_stop", MESSAGE_STOP));
	}

	@Test
	void teeEmitsVerbatimRawFramesWithEventNamesAndTypedChunksWithUsage() {
		setupChatModel(defaultFrames());

		List<DualStreamItem> items = this.chatModel
			.streamRawPassthrough(new Prompt("typed prompt"), "{\"max_tokens\":100,\"messages\":[]}")
			.collectList()
			.block();

		assertThat(items).isNotEmpty();

		// RAW branch: every frame arrives verbatim and in order - including the ping
		// event, the event: names and the unknown vendor field.
		List<DualStreamItem.RawFrame> rawFrames = items.stream()
			.filter(DualStreamItem.RawFrame.class::isInstance)
			.map(DualStreamItem.RawFrame.class::cast)
			.toList();
		assertThat(rawFrames).hasSize(7);
		assertThat(rawFrames).extracting(DualStreamItem.RawFrame::event)
			.containsExactly("message_start", "ping", "content_block_start", "content_block_delta",
					"content_block_stop", "message_delta", "message_stop");
		assertThat(rawFrames.get(0).data()).isEqualTo(MESSAGE_START);
		assertThat(rawFrames.get(0).data()).contains("\"x_vendor_extension\":\"keep-me\"");
		assertThat(rawFrames.get(1).data()).isEqualTo(PING);
		assertThat(rawFrames.get(5).data()).isEqualTo(MESSAGE_DELTA);

		// TYPED branch: the same parse pipeline as internalStream (pings dropped,
		// events assembled), producing the assistant text and the parsed usage.
		List<ChatResponse> typedResponses = items.stream()
			.filter(DualStreamItem.TypedChunk.class::isInstance)
			.map(item -> ((DualStreamItem.TypedChunk) item).response())
			.toList();
		assertThat(typedResponses).isNotEmpty();

		assertThat(typedResponses.stream()
			.anyMatch(r -> r.getResult() != null && r.getResult().getOutput().getText() != null
					&& r.getResult().getOutput().getText().contains("Hello there")))
			.isTrue();

		// The terminal message_delta merges the output tokens with the input tokens
		// remembered from message_start.
		ChatResponse lastWithUsage = typedResponses.stream()
			.filter(r -> r.getMetadata() != null && r.getMetadata().getUsage() != null
					&& r.getMetadata().getUsage().getCompletionTokens() != null
					&& r.getMetadata().getUsage().getCompletionTokens() == 12)
			.reduce((first, second) -> second)
			.orElseThrow(() -> new AssertionError("no typed chunk carried the final usage"));
		Usage usage = lastWithUsage.getMetadata().getUsage();
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
	}

	@Test
	void errorEventsAreForwardedVerbatimInRawBranch() {
		String error = "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"," + "\"message\":\"Overloaded\"}}";
		setupChatModel(
				Flux.just(sse("message_start", MESSAGE_START), sse("error", error), sse("message_stop", MESSAGE_STOP)));

		List<DualStreamItem> items = this.chatModel
			.streamRawPassthrough(new Prompt("typed prompt"), "{\"max_tokens\":100,\"messages\":[]}")
			.collectList()
			.block();

		List<DualStreamItem.RawFrame> rawFrames = items.stream()
			.filter(DualStreamItem.RawFrame.class::isInstance)
			.map(DualStreamItem.RawFrame.class::cast)
			.toList();

		// The error event must reach the raw branch verbatim (not swallowed like in
		// the typed StreamHelper path) so the client sees exactly what the provider
		// sent.
		assertThat(rawFrames).anySatisfy(frame -> {
			assertThat(frame.event()).isEqualTo("error");
			assertThat(frame.data()).isEqualTo(error);
		});
	}

	@Test
	void makesExactlyOneHttpRequest() {
		setupChatModel(defaultFrames());

		this.chatModel.streamRawPassthrough(new Prompt("typed prompt"), "{\"max_tokens\":100,\"messages\":[]}")
			.collectList()
			.block();

		verify(this.apiSpy, times(1)).chatCompletionStreamRawSse(anyString(), any());
	}

}
