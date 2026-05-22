"""
Expanded evaluation dataset (55 questions).
Categories: Math, Logic, Multi-hop, Probability, Adversarial
Difficulties: easy, medium, hard
Sources: original bench, GSM8K-style, StrategyQA-style, LogiQA-style, adversarial
"""

QUESTIONS = [
    # --- Original 10 (medium unless noted) ---
    {"id": "Q0", "category": "Math", "difficulty": "medium",
     "question": "A tank is 3/5 full. After adding 120 litres it becomes 4/5 full. What is the total capacity of the tank?",
     "answer": "600", "source": "original"},
    {"id": "Q1", "category": "Math", "difficulty": "medium",
     "question": "An investment of $5,000 grows at 8% annual interest compounded quarterly. What is the value after 3 years? Round to nearest dollar.",
     "answer": "6341", "source": "original"},
    {"id": "Q2", "category": "Math", "difficulty": "medium",
     "question": "Pipe A fills a tank in 6 hours. Pipe B drains it in 10 hours. Both open on empty tank. How many hours to fill?",
     "answer": "15", "source": "original"},
    {"id": "Q3", "category": "Logic", "difficulty": "medium",
     "question": "Alice, Bob, Carol each have cat, dog, fish (all different). Alice not cat. Bob not dog. Carol not fish. Who has which?",
     "answer": "alice fish bob cat carol dog", "source": "original"},
    {"id": "Q4", "category": "Logic", "difficulty": "medium",
     "question": "Knights always truth, knaves always lie. A says 'We are both knaves.' What are A and B?",
     "answer": "knave knight", "source": "original"},
    {"id": "Q5", "category": "Multi-hop", "difficulty": "medium",
     "question": "Eiffel Tower is in capital of France. Marie Curie did most work in this city. Name the city.",
     "answer": "paris", "source": "original"},
    {"id": "Q6", "category": "Math", "difficulty": "medium",
     "question": "Factory makes 200 widgets/day. Drops 35% for 4 days, then 110% capacity 3 days. Total widgets in 7 days?",
     "answer": "1180", "source": "original"},
    {"id": "Q7", "category": "Adversarial", "difficulty": "hard",
     "question": "A bat and ball cost $1.10. Bat costs $1 more than ball. How much is the ball?",
     "answer": "0.05", "source": "trick"},
    {"id": "Q8", "category": "Math", "difficulty": "medium",
     "question": "Train A leaves 9AM at 80 km/h from City A. Train B leaves 10AM at 100 km/h from City B (320 km away) toward A. How far from A when they meet?",
     "answer": "160", "source": "original"},
    {"id": "Q9", "category": "Probability", "difficulty": "medium",
     "question": "Bag: 4 red, 5 blue, 6 green. Two drawn without replacement. P(both same color)?",
     "answer": "34/105", "source": "original"},
    # --- Math easy ---
    {"id": "Q10", "category": "Math", "difficulty": "easy",
     "question": "What is 15 plus 23?", "answer": "38", "source": "gsm8k-style"},
    {"id": "Q11", "category": "Math", "difficulty": "easy",
     "question": "What is half of 40?", "answer": "20", "source": "gsm8k-style"},
    {"id": "Q12", "category": "Math", "difficulty": "easy",
     "question": "What is 20 percent of 100?", "answer": "20", "source": "gsm8k-style"},
    {"id": "Q13", "category": "Math", "difficulty": "easy",
     "question": "How many seconds in 3 minutes?", "answer": "180", "source": "gsm8k-style"},
    {"id": "Q14", "category": "Math", "difficulty": "easy",
     "question": "Rectangle 5m by 8m. Area in square meters?", "answer": "40", "source": "gsm8k-style"},
    # --- Math medium ---
    {"id": "Q15", "category": "Math", "difficulty": "medium",
     "question": "Buy apples 20c, sell 50c, sell 1000. Profit in dollars?", "answer": "300", "source": "gsm8k-style"},
    {"id": "Q16", "category": "Math", "difficulty": "medium",
     "question": "If 2x + 3 = 11, what is x?", "answer": "4", "source": "gsm8k-style"},
    {"id": "Q17", "category": "Math", "difficulty": "medium",
     "question": "Price $100, up 15%, then down 10%. Final price?", "answer": "103.5", "source": "gsm8k-style"},
    {"id": "Q18", "category": "Math", "difficulty": "medium",
     "question": "Scores 80 (weight 30%) and 90 (weight 70%). Weighted average?", "answer": "87", "source": "gsm8k-style"},
    {"id": "Q19", "category": "Math", "difficulty": "medium",
     "question": "Cube side length 3. Total surface area?", "answer": "54", "source": "gsm8k-style"},
    {"id": "Q20", "category": "Math", "difficulty": "medium",
     "question": "120 miles in 2h, then 180 miles in 3h. Average speed mph?", "answer": "60", "source": "gsm8k-style"},
    # --- Math hard ---
    {"id": "Q21", "category": "Math", "difficulty": "hard",
     "question": "Solve x^2 - 5x + 6 = 0. Give smaller root.", "answer": "2", "source": "gsm8k-style"},
    {"id": "Q22", "category": "Math", "difficulty": "hard",
     "question": "2x + y = 10 and x - y = 1. What is x?", "answer": "11/3", "source": "gsm8k-style"},
    {"id": "Q23", "category": "Math", "difficulty": "hard",
     "question": "How many ways to arrange 5 distinct books?", "answer": "120", "source": "gsm8k-style"},
    {"id": "Q24", "category": "Math", "difficulty": "hard",
     "question": "From 10 people choose committee of 3. How many ways?", "answer": "120", "source": "gsm8k-style"},
    {"id": "Q25", "category": "Math", "difficulty": "hard",
     "question": "Sum of first 20 odd positive integers?", "answer": "400", "source": "gsm8k-style"},
    # --- Logic ---
    {"id": "Q26", "category": "Logic", "difficulty": "easy",
     "question": "All dogs are animals. Fido is a dog. Is Fido an animal?", "answer": "yes", "source": "logiqa-style"},
    {"id": "Q27", "category": "Logic", "difficulty": "easy",
     "question": "Is 2+2=4 AND is 5>3?", "answer": "yes", "source": "logiqa-style"},
    {"id": "Q28", "category": "Logic", "difficulty": "medium",
     "question": "All cats are animals. All animals eat food. Do cats eat food?", "answer": "yes", "source": "logiqa-style"},
    {"id": "Q29", "category": "Logic", "difficulty": "medium",
     "question": "If rain then wet ground. Ground is wet. Did it necessarily rain?", "answer": "no", "source": "logiqa-style"},
    {"id": "Q30", "category": "Logic", "difficulty": "medium",
     "question": "If rain then wet. Ground not wet. Did it rain?", "answer": "no", "source": "logiqa-style"},
    {"id": "Q31", "category": "Logic", "difficulty": "hard",
     "question": "Some artists are philosophers. All philosophers are thinkers. Can we conclude some artists are thinkers?",
     "answer": "yes", "source": "logiqa-style"},
    {"id": "Q32", "category": "Logic", "difficulty": "hard",
     "question": "Statement: This sentence is false. Is it true or false?", "answer": "paradox", "source": "logiqa-style"},
    # --- Multi-hop ---
    {"id": "Q33", "category": "Multi-hop", "difficulty": "easy",
     "question": "Capital of France? In which country is the Eiffel Tower?", "answer": "france", "source": "strategyqa-style"},
    {"id": "Q34", "category": "Multi-hop", "difficulty": "medium",
     "question": "Author of Hamlet. Country of birth of that author?", "answer": "england", "source": "strategyqa-style"},
    {"id": "Q35", "category": "Multi-hop", "difficulty": "medium",
     "question": "Largest planet in solar system. How many moons does it have (approx accepted: 80+)?",
     "answer": "95", "source": "strategyqa-style"},
    {"id": "Q36", "category": "Multi-hop", "difficulty": "hard",
     "question": "Year WW2 ended. How many years after that was first Moon landing?",
     "answer": "24", "source": "strategyqa-style"},
    {"id": "Q37", "category": "Multi-hop", "difficulty": "hard",
     "question": "Inventor of telephone. Nationality of that inventor?", "answer": "scottish", "source": "strategyqa-style"},
    # --- Probability ---
    {"id": "Q38", "category": "Probability", "difficulty": "easy",
     "question": "Fair coin flipped twice. How many outcomes?", "answer": "4", "source": "original"},
    {"id": "Q39", "category": "Probability", "difficulty": "easy",
     "question": "Die rolled once. P(rolling even number)?", "answer": "1/2", "source": "original"},
    {"id": "Q40", "category": "Probability", "difficulty": "medium",
     "question": "Deck 52 cards, one drawn. P(heart)?", "answer": "1/4", "source": "original"},
    {"id": "Q41", "category": "Probability", "difficulty": "medium",
     "question": "P(rain)=0.3, P(umbrella|rain)=0.8, P(umbrella|no rain)=0.1. P(rain|umbrella)? Round 2 decimals.",
     "answer": "0.26", "source": "original"},
    {"id": "Q42", "category": "Probability", "difficulty": "hard",
     "question": "3 fair coins. P(exactly 2 heads)?", "answer": "3/8", "source": "original"},
    # --- Adversarial ---
    {"id": "Q43", "category": "Adversarial", "difficulty": "easy",
     "question": "How many animals of each kind did Moses take on the ark?", "answer": "none", "source": "adversarial"},
    {"id": "Q44", "category": "Adversarial", "difficulty": "medium",
     "question": "A clerk at a butchers shop is 5 feet 10 inches tall. What does he weigh?",
     "answer": "meat", "source": "adversarial"},
    {"id": "Q45", "category": "Adversarial", "difficulty": "medium",
     "question": "If you have only one match and enter a dark room with oil lamp, candle and fireplace, what do you light first?",
     "answer": "match", "source": "adversarial"},
    {"id": "Q46", "category": "Adversarial", "difficulty": "hard",
     "question": "Sally's mother has 4 children: April, May, June, and ?",
     "answer": "sally", "source": "adversarial"},
    {"id": "Q47", "category": "Adversarial", "difficulty": "hard",
     "question": "I have two coins totaling 30 cents. One is not a nickel. What are the coins?",
     "answer": "quarter nickel", "source": "adversarial"},
    # --- More GSM8K / mixed ---
    {"id": "Q48", "category": "Math", "difficulty": "medium",
     "question": "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many balls now?",
     "answer": "11", "source": "gsm8k-style"},
    {"id": "Q49", "category": "Math", "difficulty": "medium",
     "question": "Janet lays 16 eggs/day. Eats 3, bakes 4. Rest sold at $2 each. Daily earnings?",
     "answer": "18", "source": "gsm8k-style"},
    {"id": "Q50", "category": "Math", "difficulty": "hard",
     "question": "Bacteria double hourly. Start 100. After 8 hours count?", "answer": "25600", "source": "gsm8k-style"},
    {"id": "Q51", "category": "Logic", "difficulty": "medium",
     "question": "No C are A. All D are C. Can any D be A?", "answer": "no", "source": "logiqa-style"},
    {"id": "Q52", "category": "Multi-hop", "difficulty": "medium",
     "question": "Chemical symbol for gold. Atomic number of that element?", "answer": "79", "source": "strategyqa-style"},
    {"id": "Q53", "category": "Probability", "difficulty": "medium",
     "question": "Roll two dice. P(sum equals 7)?", "answer": "1/6", "source": "original"},
    {"id": "Q54", "category": "Adversarial", "difficulty": "medium",
     "question": "How much dirt is in a hole 2 feet by 2 feet by 2 feet?",
     "answer": "0", "source": "adversarial"},
]


def get_metadata(q_id: str) -> dict | None:
    for q in QUESTIONS:
        if q["id"] == q_id:
            return q
    return None


def filter_questions(category=None, difficulty=None, limit=None):
    out = QUESTIONS
    if category:
        out = [q for q in out if q["category"].lower() == category.lower()]
    if difficulty:
        out = [q for q in out if q["difficulty"].lower() == difficulty.lower()]
    if limit:
        out = out[:limit]
    return out
