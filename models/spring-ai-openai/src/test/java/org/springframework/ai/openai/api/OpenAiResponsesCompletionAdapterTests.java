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

package org.springframework.ai.openai.api;

import org.junit.jupiter.api.Test;

import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion.Choice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.ai.retry.NonTransientAiException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.ai.openai.api.OpenAiResponsesCompletionAdapter.toChatCompletion;

class OpenAiResponsesCompletionAdapterTests {

	@Test
	void mapsTextReasoningAndUsage() {
		ChatCompletion completion = toChatCompletion(
				"""
						{"id":"resp_1","object":"response","created_at":1700000000,"model":"gpt-test","status":"completed",
						 "output":[
						   {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Plan"},{"type":"summary_text","text":"Act"}]},
						   {"type":"message","id":"msg_1","role":"assistant","status":"completed",
						    "content":[{"type":"output_text","text":"Hel","annotations":[]},{"type":"output_text","text":"lo","annotations":[]}]}],
						 "usage":{"input_tokens":9,"input_tokens_details":{"cached_tokens":4},
						          "output_tokens":12,"output_tokens_details":{"reasoning_tokens":5},"total_tokens":21}}
						""");

		assertThat(completion.id()).isEqualTo("resp_1");
		assertThat(completion.model()).isEqualTo("gpt-test");
		assertThat(completion.created()).isEqualTo(1700000000L);
		Choice choice = completion.choices().get(0);
		assertThat(choice.finishReason()).isEqualTo(ChatCompletionFinishReason.STOP);
		assertThat(choice.message().role()).isEqualTo(Role.ASSISTANT);
		assertThat(choice.message().content()).isEqualTo("Hello");
		assertThat(choice.message().getReasoningContent()).isEqualTo("Plan\n\nAct");
		assertThat(choice.message().toolCalls()).isNull();
		assertThat(completion.usage().promptTokens()).isEqualTo(9);
		assertThat(completion.usage().completionTokens()).isEqualTo(12);
		assertThat(completion.usage().totalTokens()).isEqualTo(21);
		assertThat(completion.usage().promptTokensDetails().cachedTokens()).isEqualTo(4);
		assertThat(completion.usage().completionTokenDetails().reasoningTokens()).isEqualTo(5);
	}

	@Test
	void mapsFunctionCallsToToolCalls() {
		ChatCompletion completion = toChatCompletion(
				"""
						{"id":"resp_2","model":"gpt-test","status":"completed","output":[
						  {"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"},
						  {"type":"function_call","id":"fc_2","call_id":"call_2","name":"weather","arguments":"{\\"city\\":\\"Berlin\\"}"}]}
						""");

		Choice choice = completion.choices().get(0);
		assertThat(choice.finishReason()).isEqualTo(ChatCompletionFinishReason.TOOL_CALLS);
		assertThat(choice.message().content()).isNull();
		assertThat(choice.message().toolCalls()).extracting(ToolCall::id).containsExactly("call_1", "call_2");
		assertThat(choice.message().toolCalls().get(1).function().arguments()).isEqualTo("{\"city\":\"Berlin\"}");
		assertThat(completion.usage()).isNull();
	}

	@Test
	void mapsIncompleteResponseAndRefusal() {
		ChatCompletion completion = toChatCompletion(
				"""
						{"id":"resp_3","model":"gpt-test","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
						 "output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"No."}]}]}
						""");

		Choice choice = completion.choices().get(0);
		assertThat(choice.finishReason()).isEqualTo(ChatCompletionFinishReason.LENGTH);
		assertThat(choice.message().refusal()).isEqualTo("No.");
		assertThat(choice.message().content()).isNull();
	}

	@Test
	void raisesFailedResponse() {
		assertThatThrownBy(() -> toChatCompletion("""
				{"id":"resp_4","status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}
				""")).isInstanceOf(NonTransientAiException.class)
			.hasMessageContaining("server_error")
			.hasMessageContaining("boom");
	}

}
