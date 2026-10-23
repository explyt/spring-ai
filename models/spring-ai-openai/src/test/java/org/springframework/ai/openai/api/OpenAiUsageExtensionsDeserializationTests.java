/*
 * Copyright 2023-2026 the original author or authors.
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

import java.util.List;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.Usage;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Pins the OhMyCode gateway usage extensions ({@code compute_millis},
 * {@code wait_millis}, {@code price}, {@code price_with_discount}) on the OpenAI dialect.
 * These fields are not part of the public OpenAI API and drive billing downstream, so a
 * Jackson upgrade or an upstream merge must not drop them silently.
 */
class OpenAiUsageExtensionsDeserializationTests {

	private static final String CHUNK_WITH_NESTED_USAGE = """
			{"id":"c1","object":"chat.completion.chunk","created":1700000000,"model":"m","choices":[],
			 "usage":{"completion_tokens":5,"prompt_tokens":7,"total_tokens":12,
			  "compute_millis":321,"wait_millis":45,"price":"0.001","price_with_discount":"0.0005"}}
			""";

	// The gateway puts the extension fields at the ROOT of the streaming chunk instead
	// of inside "usage"; ChatCompletionChunk.fromJson must fold them back into Usage.
	private static final String CHUNK_WITH_MISPLACED_FIELDS = """
			{"id":"c1","object":"chat.completion.chunk","created":1700000000,"model":"m","choices":[],
			 "usage":{"completion_tokens":5,"prompt_tokens":7,"total_tokens":12},
			 "compute_millis":321,"wait_millis":45,"price":"0.001","price_with_discount":"0.0005"}
			""";

	private static final String CHUNK_WITH_MISPLACED_FIELDS_NO_USAGE = """
			{"id":"c1","object":"chat.completion.chunk","created":1700000000,"model":"m","choices":[],
			 "compute_millis":321,"price":"0.001"}
			""";

	@Test
	void chunkWithNestedUsageKeepsExtensionFields() {
		ChatCompletionChunk chunk = ModelOptionsUtils.jsonToObject(CHUNK_WITH_NESTED_USAGE, ChatCompletionChunk.class);
		assertExtensions(chunk.usage());
	}

	@Test
	void misplacedRootFieldsAreFoldedIntoUsage() {
		ChatCompletionChunk chunk = ModelOptionsUtils.jsonToObject(CHUNK_WITH_MISPLACED_FIELDS,
				ChatCompletionChunk.class);
		assertExtensions(chunk.usage());
		assertThat(chunk.usage().completionTokens()).isEqualTo(5);
		assertThat(chunk.usage().promptTokens()).isEqualTo(7);
		assertThat(chunk.usage().totalTokens()).isEqualTo(12);
	}

	@Test
	void misplacedRootFieldsWithoutUsageAreIgnored() {
		ChatCompletionChunk chunk = ModelOptionsUtils.jsonToObject(CHUNK_WITH_MISPLACED_FIELDS_NO_USAGE,
				ChatCompletionChunk.class);
		assertThat(chunk.id()).isEqualTo("c1");
		assertThat(chunk.usage()).isNull();
	}

	@Test
	void streamingParserFoldsMisplacedFields() {
		OpenAiApi api = OpenAiApi.builder().apiKey("test").build();
		List<ChatCompletionChunk> chunks = api
			.parseChatCompletionChunks(Flux.just(CHUNK_WITH_MISPLACED_FIELDS, "[DONE]"))
			.collectList()
			.block();
		assertThat(chunks).hasSize(1);
		assertExtensions(chunks.get(0).usage());
	}

	@Test
	void nonStreamingCompletionKeepsExtensionFields() {
		String json = """
				{"id":"c1","object":"chat.completion","created":1700000000,"model":"m","choices":[],
				 "usage":{"completion_tokens":5,"prompt_tokens":7,"total_tokens":12,
				  "compute_millis":321,"wait_millis":45,"price":"0.001","price_with_discount":"0.0005"}}
				""";
		ChatCompletion completion = ModelOptionsUtils.jsonToObject(json, ChatCompletion.class);
		assertExtensions(completion.usage());
	}

	private static void assertExtensions(Usage usage) {
		assertThat(usage).isNotNull();
		assertThat(usage.computeMillis()).isEqualTo(321L);
		assertThat(usage.waitMillis()).isEqualTo(45L);
		assertThat(usage.price()).isEqualTo("0.001");
		assertThat(usage.priceWithDiscount()).isEqualTo("0.0005");
	}

}
