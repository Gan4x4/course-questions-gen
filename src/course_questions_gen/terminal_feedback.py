"""Terminal feedback helpers."""


def collect_feedback_from_terminal(payload):
    approved = {}

    for topic, questions in payload["topics"].items():
        print("\nTopic:", topic)

        if not questions:
            print("No questions to review.")
            approved[topic] = []
            continue

        for index, question in enumerate(questions, start=1):
            print()
            print(f"{question['number']}.", question["question"])
            print("Answer:", question.get("answer", ""))

        approved[topic] = ask_approved_questions(questions)

    return approved


def ask_approved_questions(questions):
    while True:
        answer = input("Approve numbers, Y for all, N for none: ").strip()
        selected_questions = select_questions(answer, questions)

        if selected_questions is not None:
            return selected_questions

        print("Enter Y, N, or question numbers like 1 3.")


def select_questions(answer: str, questions):
    normalized_answer = answer.lower()

    if normalized_answer == "y":
        return list(range(1, len(questions) + 1))

    if normalized_answer == "n":
        return []

    indexes = parse_question_numbers(answer, len(questions))
    if indexes is None:
        return None

    return indexes


def parse_question_numbers(answer: str, question_count: int):
    parts = answer.replace(",", " ").split()
    if not parts:
        return None

    indexes = []
    for part in parts:
        if not part.isdigit():
            return None

        index = int(part)
        if index < 1 or index > question_count:
            return None

        if index not in indexes:
            indexes.append(index)

    return indexes
