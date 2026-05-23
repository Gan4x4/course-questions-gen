You are creating content expert agents for generating student questions.

Create exactly one expert for each topic listed below.

Section: {section}

Topics:
{topics}

For each expert:
- set `topic` to exactly one topic from the list
- do not modify, expand, rename, or combine the `topic` value
- write a concise `description` in plain English
- in `description`, expand the topic with relevant subtopics, edge cases, tiny implementation details, and common student misunderstandings but avoid intersection with other topics
- keep each expert's focus distinct; avoid overlap with other listed topics
- make the description specific to the `topic` and `section`
- focus on practical student-question generation, not broad theory

Do not add experts for topics that are not listed.
Do not skip any listed topic.
