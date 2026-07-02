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

package org.springframework.ai.openai.chat;

import java.util.List;

import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk.ChunkChoice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.StringUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.BDDMockito.given;

/**
 * Unit test for the OpenAI raw-passthrough streaming path ({@code streamRaw}).
 *
 * <p>
 * Regression guard for the streaming-usage accounting bug: on the raw path
 * {@code applyOverrides} forces {@code include_usage=true} onto the wire body, but the
 * typed {@code ChatCompletionRequest} it derives from has no {@code streamOptions}. The
 * usage-merge gate must therefore key off the raw body, not the typed request, or the
 * final usage chunk is dropped and token totals come out empty.
 */
@ExtendWith(MockitoExtension.class)
public class OpenAiChatModelRawPassthroughTests {

	@Mock
	private OpenAiApi openAiApi;

	private OpenAiChatModel chatModel;

	private void setupChatModel() {
		RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;
		ToolCallingManager toolCallingManager = ToolCallingManager.builder().build();
		ObservationRegistry observationRegistry = ObservationRegistry.NOOP;
		// Default options carry no streamOptions, mirroring how ai-router drives the raw
		// path: include_usage lives only in the raw body, never in the typed request.
		this.chatModel = new OpenAiChatModel(this.openAiApi, OpenAiChatOptions.builder().build(), toolCallingManager,
				retryTemplate, observationRegistry);
	}

	@Test
	void streamRawAccumulatesUsageEvenWhenTypedStreamOptionsAreAbsent() {
		setupChatModel();

		// A content chunk (no usage) followed by the final usage-only chunk OpenAI emits
		// when stream_options.include_usage=true — exactly what applyOverrides forces.
		var contentChoice = new ChunkChoice(null, 0, new ChatCompletionMessage("Hello", Role.ASSISTANT), null);
		ChatCompletionChunk contentChunk = new ChatCompletionChunk("id", List.of(contentChoice), 666L, "model", null,
				null, "chat.completion.chunk", null);
		ChatCompletionChunk usageChunk = new ChatCompletionChunk("id", List.of(), 666L, "model", null, null,
				"chat.completion.chunk", new OpenAiApi.Usage(12, 9, 21));

		given(this.openAiApi.chatCompletionStreamRaw(anyString(), any()))
			.willReturn(Flux.just(contentChunk, usageChunk));

		// A minimal valid JSON object is enough — applyOverrides only needs to parse it.
		List<ChatResponse> responses = this.chatModel.streamRaw(new Prompt("test"), "{\"messages\":[]}")
			.collectList()
			.block();

		assertThat(responses).isNotEmpty();

		// The usage from the final chunk must be merged onto the content response, just
		// like the typed streaming path. Before the fix the gate read the (null) typed
		// streamOptions, so the content response carried an empty usage.
		ChatResponse contentResponse = responses.stream()
			.filter(r -> r.getResult() != null && StringUtils.hasText(r.getResult().getOutput().getText()))
			.findFirst()
			.orElseThrow(() -> new AssertionError("no response carried the assistant text"));

		Usage usage = contentResponse.getMetadata().getUsage();
		assertThat(usage).isNotNull();
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
		assertThat(usage.getTotalTokens()).isEqualTo(21);
	}

}
