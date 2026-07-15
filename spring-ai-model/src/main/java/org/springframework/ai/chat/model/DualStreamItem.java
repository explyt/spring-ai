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

package org.springframework.ai.chat.model;

/**
 * "Dual" = each element is EITHER the typed parse ({@link TypedChunk}, for usage/audit)
 * OR the verbatim raw provider SSE frame ({@link RawFrame}, to relay), from one teed
 * stream.
 *
 * @since 1.1.7
 */
public sealed interface DualStreamItem permits DualStreamItem.TypedChunk, DualStreamItem.RawFrame {

	/**
	 * The typed parse of a provider chunk, identical to what the normal streaming path
	 * emits.
	 *
	 * @param response the typed chat response
	 */
	record TypedChunk(ChatResponse response) implements DualStreamItem {
	}

	/**
	 * A raw provider SSE frame forwarded verbatim.
	 *
	 * @param event the SSE {@code event:} name, or {@code null} for the OpenAI chat
	 * dialect (which does not use event names)
	 * @param data the SSE frame body verbatim, including sentinels such as {@code [DONE]}
	 * and usage frames
	 */
	record RawFrame(String event, String data) implements DualStreamItem {
	}

}
