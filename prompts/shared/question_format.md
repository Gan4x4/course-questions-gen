# Question Format

Generate questions in English, using the compact Q/A style from the examples.

Each question must be:
- narrow, short, and specific
- about one concrete concept, API, design choice, or failure mode
- answerable in one or two short sentences
- suitable for checking practical understanding, not broad essay knowledge

Each answer must be:
- as short as possible
- technically correct
- direct, without long explanations
- preferably one sentence or a compact list

Avoid:
- broad questions like "What is LangGraph?"
- vague questions like "Explain agents"
- multi-part questions unless the expected answer is still very short
- open discussion prompts
- answers with unnecessary background context

Preferred row shape:

Proof

- All question must contain 1-3 valid url of artile with answer or explanation
- Avoid long articles prefer short
- Use html anchors for part of article related to question
- Try to use all 3 links


```csv
Section,Subsection,Question,Answer,Link1,Link2,Link3,Notes
LangGraph,Graph basics,What is a node in LangGraph?,A function or runnable that receives state and returns a state update.,,,,
LangGraph,Conditional routing,What does conditional routing choose?,The next node based on the current state.,,,,
```
