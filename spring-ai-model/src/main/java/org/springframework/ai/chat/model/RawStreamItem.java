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
 * Item emitted by a raw-response passthrough stream. A single provider HTTP stream is
 * "teed" into two logical streams that share one upstream subscription:
 * <ul>
 * <li>{@link TypedChunk} — the typed {@link ChatResponse} produced by the exact same
 * parsing pipeline used by the normal streaming path (for usage accounting and audit);
 * <li>{@link RawFrame} — the raw provider SSE frame, forwarded verbatim, so the caller
 * can relay it to its own client without lossy re-serialization.
 * </ul>
 *
 * <p>
 * This sealed type intentionally forces the (single) consumer to handle both variants and
 * prevents the raw variant from leaking into the existing typed operators.
 *
 * @since 1.1.7
 */
public sealed interface RawStreamItem permits RawStreamItem.TypedChunk, RawStreamItem.RawFrame {

	/**
	 * The typed parse of a provider chunk, identical to what the normal streaming path
	 * emits.
	 *
	 * @param response the typed chat response
	 */
	record TypedChunk(ChatResponse response) implements RawStreamItem {
	}

	/**
	 * A raw provider SSE frame forwarded verbatim.
	 *
	 * @param event the SSE {@code event:} name, or {@code null} for the OpenAI chat
	 * dialect (which does not use event names)
	 * @param data the SSE frame body verbatim, including sentinels such as {@code [DONE]}
	 * and usage frames
	 */
	record RawFrame(String event, String data) implements RawStreamItem {
	}

}
