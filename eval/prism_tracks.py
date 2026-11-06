"""PRISM-Bench alignment tracks and the request classifier prompt (gpt-6.1-sol, strict JSON). Used by prism_score.py classify."""
TRACKS = ["text_rendering", "entity", "style", "affection", "composition", "imagination", "long_text"]
GUIDE = """PRISM-Bench scores how well a generated image follows its text prompt with one of seven track-specific rubrics.
Each rubric looks for different failures:
- text_rendering: checks that requested words appear exactly (spelling, wording) and are legible and well placed. Spelling errors cost the most. Ignores most other content.
- entity: checks that a specific named real-world entity (a real person, brand, logo, landmark, named character, artwork, product model, species) is recognisable and accurate, in the right scene.
- style: checks that a requested art style, medium, artist, era or aesthetic movement is executed with real understanding, not as a filter.
- affection: checks the literal content first, then whether the requested emotion or mood is conveyed with nuance.
- composition: checks object counts, spatial relations (left of, on top of, in a grid), attributes (colour, size) and interactions between several objects.
- imagination: checks that an invented, fantastical or physically impossible object or creature combines its described features into one coherent whole.
- long_text: checks coverage of a long, detailed description: major elements, then every secondary detail and attribute, one point per omission.

Assign the user request below to the ONE track whose rubric best captures what the request most needs to get right, so
that the score reflects success on this request. Rules:
- text_rendering when the request names words that must appear on the image (a title, name, slogan, date, list items,
  slide text, quoted strings) and getting those words right is central, even if other content is also requested.
  Generic "add a title" without the words does not count.
- entity only when a specific named entity must be recognisable; generic people or objects are not entities.
- style only when the request asks for a specific style or medium and that style is a main point, not a side note
  such as "realistic photo" or "modern design".
- affection only when conveying an emotion or mood is the main point.
- composition when the request is mainly about several objects with counts, positions, arrangement or attributes.
- imagination when the subject is invented or impossible.
- long_text when the request is a long, detailed, multi-part description with no single dominant aspect above, or
  when it is a short generic request with no specific emphasis.
Edit-style requests ("change the background to ...") are classified by what the requested result needs.
Also list the other tracks that clearly apply (secondary, possibly empty), a confidence, and a one-sentence reason in
Simplified Chinese (at most 50 characters)."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["primary", "secondary", "confidence", "reason_zh"],
          "properties": {"primary": {"type": "string", "enum": TRACKS},
                         "secondary": {"type": "array", "items": {"type": "string", "enum": TRACKS}},
                         "confidence": {"type": "string", "enum": ["high", "medium", "low"]}, "reason_zh": {"type": "string"}}}
