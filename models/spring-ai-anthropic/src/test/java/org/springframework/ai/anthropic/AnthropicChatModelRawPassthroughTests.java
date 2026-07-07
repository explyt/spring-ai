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

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Flux;

import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.anthropic.api.AnthropicApi.ChatCompletionResponse;
import org.springframework.ai.anthropic.api.AnthropicApi.ContentBlock;
import org.springframework.ai.anthropic.api.AnthropicApi.Role;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;

/**
 * Unit tests for the Anthropic raw-passthrough path ({@code callRaw} /
 * {@code streamRaw}): the raw body must be forwarded with only {@code model} and
 * {@code stream} overridden, {@code max_tokens} injected only when absent, and every
 * other field (including vendor extensions) preserved verbatim.
 */
@ExtendWith(MockitoExtension.class)
public class AnthropicChatModelRawPassthroughTests {

	private static final String MODEL = "claude-sonnet-4-5";

	@Mock
	private AnthropicApi anthropicApi;

	private AnthropicChatModel chatModel;

	private void setupChatModel() {
		this.chatModel = AnthropicChatModel.builder()
			.anthropicApi(this.anthropicApi)
			.defaultOptions(AnthropicChatOptions.builder().model(MODEL).maxTokens(4096).build())
			.build();
	}

	private static ChatCompletionResponse completionResponse() {
		return new ChatCompletionResponse("msg_1", "message", Role.ASSISTANT, List.of(new ContentBlock("Hello there")),
				MODEL, "end_turn", null, new AnthropicApi.Usage(9, 12, null, null));
	}

	@Test
	void callRawForwardsBodyWithOverridesAndPreservesClientFields() throws Exception {
		setupChatModel();

		ArgumentCaptor<String> bodyCaptor = ArgumentCaptor.forClass(String.class);
		given(this.anthropicApi.chatCompletionEntityRaw(bodyCaptor.capture(), any()))
			.willReturn(ResponseEntity.ok(completionResponse()));

		String rawBody = """
				{"model":"client-model","max_tokens":777,"stream":true,
				 "messages":[{"role":"user","content":"raw hello"}],
				 "vendor_extension":{"keep":"me"}}""";

		ChatResponse response = this.chatModel.callRaw(new Prompt("typed prompt"), rawBody);

		JsonNode sent = ModelOptionsUtils.OBJECT_MAPPER.readTree(bodyCaptor.getValue());
		// Gateway-owned overrides.
		assertThat(sent.get("model").asText()).isEqualTo(MODEL);
		assertThat(sent.get("stream").asBoolean()).isFalse();
		// Client fields preserved verbatim.
		assertThat(sent.get("max_tokens").asInt()).isEqualTo(777);
		assertThat(sent.get("messages").get(0).get("content").asText()).isEqualTo("raw hello");
		assertThat(sent.get("vendor_extension").get("keep").asText()).isEqualTo("me");

		assertThat(response.getResult().getOutput().getText()).isEqualTo("Hello there");
		Usage usage = response.getMetadata().getUsage();
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
	}

	@Test
	void callRawInjectsMaxTokensOnlyWhenAbsent() throws Exception {
		setupChatModel();

		ArgumentCaptor<String> bodyCaptor = ArgumentCaptor.forClass(String.class);
		given(this.anthropicApi.chatCompletionEntityRaw(bodyCaptor.capture(), any()))
			.willReturn(ResponseEntity.ok(completionResponse()));

		this.chatModel.callRaw(new Prompt("typed prompt"), "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");

		JsonNode sent = ModelOptionsUtils.OBJECT_MAPPER.readTree(bodyCaptor.getValue());
		// Anthropic requires max_tokens; the gateway default fills the gap.
		assertThat(sent.get("max_tokens").asInt()).isEqualTo(4096);
	}

	@Test
	void streamRawForwardsBodyWithStreamTrueAndParsesUsage() throws Exception {
		setupChatModel();

		ArgumentCaptor<String> bodyCaptor = ArgumentCaptor.forClass(String.class);
		given(this.anthropicApi.chatCompletionStreamRaw(bodyCaptor.capture(), any()))
			.willReturn(Flux.just(completionResponse()));

		String rawBody = "{\"model\":\"client-model\",\"max_tokens\":100,\"messages\":[]}";
		List<ChatResponse> responses = this.chatModel.streamRaw(new Prompt("typed prompt"), rawBody)
			.collectList()
			.block();

		JsonNode sent = ModelOptionsUtils.OBJECT_MAPPER.readTree(bodyCaptor.getValue());
		assertThat(sent.get("model").asText()).isEqualTo(MODEL);
		assertThat(sent.get("stream").asBoolean()).isTrue();
		assertThat(sent.get("max_tokens").asInt()).isEqualTo(100);

		assertThat(responses).isNotEmpty();
		ChatResponse last = responses.get(responses.size() - 1);
		Usage usage = last.getMetadata().getUsage();
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
	}

	@Test
	void callRawRejectsNonObjectBody() {
		setupChatModel();

		assertThatThrownBy(() -> this.chatModel.callRaw(new Prompt("typed prompt"), "[1,2,3]"))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("JSON object");
	}

	@Test
	void callRawRejectsMalformedBody() {
		setupChatModel();

		assertThatThrownBy(() -> this.chatModel.callRaw(new Prompt("typed prompt"), "{not json"))
			.isInstanceOf(IllegalArgumentException.class);
	}

}
