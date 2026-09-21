"""Builds a deterministic, deliberately hard retrieval benchmark (fictional Acme Corp handbook).

Run:  python build_benchmark.py        (writes ./docs and ./questions.jsonl; same output every time)

The corpus is designed so that different techniques earn their keep:

  country      eight near-identical country supplements. Dense embeddings blur "Germany" and "France";
               exact-term search does not.                                       -> hybrid search
  exact-id     form / policy codes ("HR-311"). Embeddings have no idea what a code means.  -> hybrid
  vocabulary   everyday questions against formal policy wording ("my father passed away" vs
               "compassionate absence").                                    -> reranking, query rewriting
  table        facts that live in table rows.                                     -> table-aware chunking
  buried       a specific fact deep inside a long section.                        -> parent-child chunking
  followup     "And in France?" needs the earlier question.                       -> follow-up condensation

Every answerable question carries a gold *quote*, so a retrieval "hit" means the returned passage
actually contains the answer text, not merely that it came from the right document.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
rng = random.Random(7)

docs: dict[str, str] = {}
questions: list[dict] = []


def add_question(qid, question, doc, section, quote, answer, tag, history=None):
    case = {
        "id": qid,
        "question": question,
        "gold_answer": answer,
        "gold_sources": [{"source": doc, "section": section, "quote": quote}],
        "tags": [tag],
        "split": "test" if len(questions) % 2 else "dev",
    }
    if history:
        case["history"] = history
    questions.append(case)


# -- country supplements ------------------------------------------------------------------------

COUNTRIES = [
    "United States",
    "United Kingdom",
    "Germany",
    "France",
    "India",
    "Singapore",
    "Brazil",
    "Japan",
]
country_facts: dict[str, dict[str, int]] = {}
for country in COUNTRIES:
    f = {
        "leave": rng.randint(18, 30),
        "advance": rng.randint(3, 10),
        "holidays": rng.randint(8, 16),
        "comp": rng.randint(2, 8),
        "notice": rng.choice([14, 30, 45, 60, 90]),
        "probation": rng.choice([3, 4, 6]),
        "probation_notice": rng.choice([7, 14, 15]),
        "overtime_x": rng.choice([1.25, 1.5, 2.0]),
        "overtime_cap": rng.randint(20, 60),
        "parental": rng.randint(12, 52),
        "secondary": rng.randint(1, 10),
        "cover": rng.choice([50, 75, 100, 150, 250]) * 1000,
        "remote": rng.randint(1, 3),
    }
    country_facts[country] = f
    slug = country.lower().replace(" ", "-")
    docs[f"supplement-{slug}.md"] = f"""# {country} Country Supplement

This supplement sets out the terms that apply to employees based in {country}. It should be read
together with the global handbook.

## Annual leave

Full-time employees in {country} are entitled to {f["leave"]} days of paid annual leave per calendar year. Leave accrues monthly and requests are made through the HR portal at least {f["advance"]} working days in advance.

## Public holidays

{country} follows a local calendar of {f["holidays"]} paid public holidays. Employees required to work on a public holiday receive a compensatory day off within {f["comp"]} weeks.

## Notice period

After probation, either party must give {f["notice"]} days written notice of resignation or termination in {country}.

## Probation

New hires in {country} serve a probation period of {f["probation"]} months, during which the notice period is {f["probation_notice"]} days.

## Overtime

Approved overtime in {country} is paid at {f["overtime_x"]} times the base hourly rate. Overtime above {f["overtime_cap"]} hours per month needs director approval.

## Parental leave

Primary caregivers in {country} receive {f["parental"]} weeks of paid parental leave, and secondary caregivers receive {f["secondary"]} weeks.

## Health insurance

The company health plan in {country} covers employees and includes an annual limit of {f["cover"]:,} dollars.

## Remote work

Employees in {country} may work remotely up to {f["remote"]} days per week after completing probation.
"""

country_questions = [
    (
        "Annual leave",
        "How many days of annual leave do employees in {c} get?",
        "leave",
        "{leave} days of paid annual leave",
        "{leave} days",
    ),
    (
        "Notice period",
        "What notice period applies after probation in {c}?",
        "notice",
        "{notice} days written notice",
        "{notice} days",
    ),
    (
        "Probation",
        "How long is the probation period in {c}?",
        "probation",
        "probation period of {probation} months",
        "{probation} months",
    ),
    (
        "Overtime",
        "What rate is overtime paid at in {c}?",
        "overtime_x",
        "paid at {overtime_x} times the base hourly rate",
        "{overtime_x} times the base rate",
    ),
    (
        "Parental leave",
        "How many weeks of parental leave do primary caregivers get in {c}?",
        "parental",
        "{parental} weeks of paid parental leave",
        "{parental} weeks",
    ),
    (
        "Remote work",
        "How many days per week can people in {c} work remotely?",
        "remote",
        "up to {remote} days per week",
        "{remote} days per week",
    ),
    (
        "Public holidays",
        "How many paid public holidays does {c} have?",
        "holidays",
        "{holidays} paid public holidays",
        "{holidays} public holidays",
    ),
    (
        "Health insurance",
        "What is the annual health insurance limit in {c}?",
        "cover",
        "annual limit of {cover_fmt} dollars",
        "{cover_fmt} dollars",
    ),
]
for index, country in enumerate(COUNTRIES):
    slug = country.lower().replace(" ", "-")
    facts = {**country_facts[country], "cover_fmt": f"{country_facts[country]['cover']:,}"}
    for offset in range(2):
        section, question, _, quote, answer = country_questions[
            (index * 2 + offset) % len(country_questions)
        ]
        add_question(
            f"country-{index}{offset}",
            question.format(c=country),
            f"supplement-{slug}.md",
            section,
            quote.format(**facts),
            answer.format(**facts),
            "country",
        )

# -- forms and codes ----------------------------------------------------------------------------

FORMS = [
    (
        "HR-104",
        "Change of personal details",
        "update your home address, bank account or emergency contact",
        "Changes take effect from the next payroll run",
    ),
    (
        "HR-118",
        "Leave encashment",
        "convert unused earned leave into pay at year end",
        "Requests must reach payroll before 15 December",
    ),
    (
        "HR-207",
        "Parental leave extension",
        "extend parental leave beyond the standard entitlement",
        "Submit it four weeks before your leave is due to end",
    ),
    (
        "HR-215",
        "Return to work after long leave",
        "arrange a phased return after an absence of more than three months",
        "Your manager and an HR partner must both sign it",
    ),
    (
        "HR-233",
        "Flexible working arrangement",
        "propose a permanent change to your working pattern",
        "Decisions are made within thirty days",
    ),
    (
        "HR-256",
        "Internal transfer request",
        "apply to move to a different team or office",
        "Your current manager is informed only after the receiving team agrees",
    ),
    (
        "HR-274",
        "Relocation assistance claim",
        "claim moving costs when the company relocates you",
        "Original invoices must be attached",
    ),
    (
        "HR-289",
        "Tuition reimbursement claim",
        "claim back the cost of approved job-related courses",
        "Attach proof of successful completion",
    ),
    (
        "HR-301",
        "Grievance submission",
        "raise a formal complaint about your treatment at work",
        "People Operations acknowledges receipt within two working days",
    ),
    (
        "HR-311",
        "Sabbatical request",
        "request an unpaid sabbatical of between one and six months",
        "Submit it at least sixty days before the intended start date",
    ),
    (
        "HR-327",
        "Resignation notice",
        "formally give notice that you are leaving the company",
        "Your notice period starts on the date HR stamps the form",
    ),
    (
        "HR-342",
        "Exit clearance",
        "confirm you have returned equipment and settled advances",
        "Final pay is released only after clearance is complete",
    ),
    (
        "HR-358",
        "Referral bonus claim",
        "claim a bonus for a successful employee referral",
        "The referred hire must complete six months of service first",
    ),
    (
        "HR-366",
        "Equipment return",
        "return a laptop or phone to the IT service desk",
        "A receipt is issued for every returned item",
    ),
    (
        "HR-371",
        "Secondment application",
        "apply for a temporary assignment with a partner organisation",
        "Secondments last between three and twelve months",
    ),
    (
        "HR-388",
        "Performance review appeal",
        "challenge the outcome of your annual performance review",
        "Appeals must be filed within ten working days of the review",
    ),
    (
        "HR-402",
        "Wellness stipend claim",
        "claim your monthly wellness stipend",
        "Claims are paid together with your salary",
    ),
    (
        "HR-419",
        "Dependent enrolment",
        "add a spouse or child to your health insurance",
        "Enrolment is possible within thirty days of a life event",
    ),
    (
        "HR-433",
        "Conflict of interest declaration",
        "declare a personal interest that could affect your work",
        "Declarations are reviewed by the Compliance team",
    ),
    (
        "HR-447",
        "Remote work equipment allowance",
        "claim the one-time allowance for a desk or chair",
        "The allowance is limited to three hundred dollars",
    ),
]
forms_text = "# HR Forms Catalogue\n\nEvery HR request starts with a numbered form. Find the form below and send it to People Operations.\n"
for code, title, purpose, note in FORMS:
    forms_text += f"\n### Form {code}: {title}\n\nUse form {code} to {purpose}. {note}.\n"
docs["forms-catalogue.md"] = forms_text

for index, (code, _title, purpose, _note) in enumerate(rng.sample(FORMS, 10)):
    add_question(
        f"exactid-{index}",
        f"What is form {code} for?",
        "forms-catalogue.md",
        f"Form {code}",
        f"Use form {code} to {purpose}",
        purpose,
        "exact-id",
    )
for index, (code, _title, purpose, _note) in enumerate(rng.sample(FORMS, 5)):
    add_question(
        f"purpose-{index}",
        f"Which form do I use to {purpose}?",
        "forms-catalogue.md",
        f"Form {code}",
        f"Use form {code} to {purpose}",
        f"Form {code}",
        "vocabulary",
    )

# -- formal wording vs everyday questions -------------------------------------------------------

POLICIES = [
    (
        "Compassionate absence",
        "Employees are entitled to five working days of compassionate absence following the death of an immediate family member. Additional unpaid absence may be agreed with line management.",
        "My father passed away. How much time off can I take?",
        "five working days",
    ),
    (
        "Civic duty absence",
        "An employee summoned to serve on a jury shall be released from duty with full pay for the duration of that service, up to twenty working days.",
        "I got called up for jury service. Will I still get paid?",
        "full pay",
    ),
    (
        "Workplace relocation",
        "Where the company requires an employee to change their permanent work location by more than fifty kilometres, a relocation package of up to eight thousand dollars is available.",
        "The company is moving me to another city. Do they cover the cost of moving?",
        "eight thousand dollars",
    ),
    (
        "Continuing education",
        "The company reimburses up to three thousand dollars per calendar year for approved job-related coursework upon successful completion.",
        "Will the company pay for me to do an online certification?",
        "three thousand dollars",
    ),
    (
        "Wellbeing allowance",
        "A wellbeing allowance of forty dollars per month may be claimed towards fitness memberships or mindfulness subscriptions.",
        "Can I get money back for my gym membership?",
        "forty dollars per month",
    ),
    (
        "Formal complaints",
        "An employee wishing to raise a formal complaint about their treatment at work should submit a written grievance to People Operations, which will respond within ten working days.",
        "Someone at work is treating me unfairly. Who do I complain to and how quickly will they reply?",
        "ten working days",
    ),
    (
        "Speaking up",
        "Concerns about financial impropriety may be reported anonymously to the Audit Committee hotline; retaliation against anyone who reports in good faith is strictly prohibited.",
        "I think someone is fiddling the accounts. Can I report it without giving my name?",
        "reported anonymously to the Audit Committee hotline",
    ),
    (
        "Personal data incidents",
        "Any suspected loss of personal data must be notified to the Data Protection Officer within twenty-four hours of discovery.",
        "I accidentally emailed a customer list to the wrong person. How fast do I have to tell someone?",
        "within twenty-four hours",
    ),
    (
        "Lost or stolen devices",
        "Loss or theft of a company-issued device must be reported to the IT Service Desk immediately, and to the police where theft is suspected.",
        "My work phone got stolen on the train. What should I do?",
        "reported to the IT Service Desk immediately",
    ),
    (
        "Carer's absence",
        "Employees who provide unpaid care for a dependent relative may request up to ten days per year of carer's absence.",
        "I sometimes need to look after my elderly mother. Is there any leave for that?",
        "ten days per year of carer's absence",
    ),
    (
        "Charitable volunteering",
        "Two paid days per year are provided for approved charitable volunteering.",
        "Can I take paid days off to volunteer at an animal shelter?",
        "Two paid days per year",
    ),
    (
        "Core hours",
        "Core collaboration hours are 10:00 to 15:00 local time; outside these hours employees may schedule their work flexibly.",
        "What hours do I actually have to be online?",
        "10:00 to 15:00 local time",
    ),
    (
        "Client dress standards",
        "Business casual attire is expected in client-facing settings.",
        "Do I have to wear a suit when I meet customers?",
        "Business casual attire",
    ),
    (
        "Business travel cover",
        "Business travellers are covered by the corporate travel insurance policy, including medical evacuation.",
        "What happens if I fall ill on a work trip abroad?",
        "including medical evacuation",
    ),
    (
        "Termination of employment",
        "Employees wishing to terminate their contract must provide written notice to their line manager and to People Operations.",
        "How do I quit?",
        "written notice to their line manager and to People Operations",
    ),
]
docs["people-policies.md"] = (
    "# People Policies\n\nThe following policies apply to all employees.\n"
    + "".join(f"\n## {heading}\n\n{text}\n" for heading, text, _, _ in POLICIES)
)
for index, (heading, _text, question, answer) in enumerate(POLICIES):
    add_question(
        f"vocab-{index}", question, "people-policies.md", heading, answer, answer, "vocabulary"
    )

# -- tables ---------------------------------------------------------------------------------------

grades = []
low = 42000
for level in range(1, 8):
    high = low + rng.randint(14, 26) * 1000
    grades.append((f"L{level}", low, high))
    low = high - rng.randint(4, 9) * 1000
salary = "# Salary Bands\n\nBands are annual base salary in US dollars.\n\n| Grade | Minimum | Maximum |\n| --- | --- | --- |\n"
salary += "".join(f"| {g} | {lo:,} | {hi:,} |\n" for g, lo, hi in grades)
docs["salary-bands.md"] = salary
docs["benefits-comparison.md"] = """# Benefits Comparison

## Health plans

| Plan | Monthly cost | Dependants covered | Dental | Annual limit |
| --- | --- | --- | --- | --- |
| Basic | 0 | None | No | 100,000 |
| Plus | 40 | Spouse | Yes | 250,000 |
| Family | 90 | Spouse and children | Yes | 500,000 |

## Pension

| Service | Company match |
| --- | --- |
| Under 2 years | 3 percent |
| 2 to 5 years | 5 percent |
| Over 5 years | 8 percent |
"""
for index, (grade, lo, hi) in enumerate(rng.sample(grades, 3)):
    add_question(
        f"table-{index}",
        f"What is the maximum salary for grade {grade}?",
        "salary-bands.md",
        "Salary Bands",
        f"| {grade} | {lo:,} | {hi:,} |",
        f"{hi:,} dollars",
        "table",
    )
add_question(
    "table-3",
    "Which health plan covers my spouse and children, and what does it cost per month?",
    "benefits-comparison.md",
    "Health plans",
    "| Family | 90 | Spouse and children |",
    "The Family plan, 90 dollars per month",
    "table",
)
add_question(
    "table-4",
    "How much does the company match on my pension if I have worked here for three years?",
    "benefits-comparison.md",
    "Pension",
    "| 2 to 5 years | 5 percent |",
    "5 percent",
    "table",
)
add_question(
    "table-5",
    "Does the Basic health plan include dental?",
    "benefits-comparison.md",
    "Health plans",
    "| Basic | 0 | None | No |",
    "No",
    "table",
)

# -- a long handbook with facts buried inside sections --------------------------------------------

FILLER = [
    "All staff must follow the documented procedure and record each action in the ticketing system.",
    "Line managers are responsible for making sure their teams understand and apply this guidance.",
    "Exceptions require written approval and must be reviewed at the next quarterly governance meeting.",
    "Where several teams are affected, the coordinating team keeps everyone informed of progress.",
    "Records are retained for the period set out in the retention schedule and then securely destroyed.",
    "Colleagues are encouraged to raise improvement ideas with the security governance forum.",
    "This guidance is reviewed every year and whenever there is a significant change in the environment.",
]
BURIED = {
    "Incident response": (
        "Outside working hours the on-call security officer is reached on extension 5550142.",
        "Who do I call for a security incident at 2am?",
        "extension 5550142",
    ),
    "Access management": (
        "Contractors receive access for a maximum of ninety days before it must be renewed.",
        "How long can a contractor's system access last before renewal?",
        "maximum of ninety days",
    ),
    "Device policy": (
        "Personal phones may only connect to the guest wireless network, never to the corporate network.",
        "Can I connect my own phone to the office wifi?",
        "guest wireless network",
    ),
    "Data classification": (
        "Documents marked Restricted must never be sent to a personal email address.",
        "Can I forward a Restricted document to my personal email?",
        "never be sent to a personal email address",
    ),
    "Vendor security": (
        "New vendors that handle customer data must complete the security questionnaire before any contract is signed.",
        "What must a new vendor complete before we sign a contract with them?",
        "security questionnaire before any contract is signed",
    ),
    "Physical security": (
        "Visitors must wear a red lanyard and be escorted at all times.",
        "What colour lanyard do visitors wear?",
        "red lanyard",
    ),
}
security = "# Security Handbook\n\nThis handbook describes how the company protects its people, information and premises.\n"
for heading, (fact, _, _) in BURIED.items():
    sentences = [rng.choice(FILLER) for _ in range(9)]
    sentences.insert(rng.randint(4, 6), fact)
    security += f"\n## {heading}\n\n" + " ".join(sentences) + "\n"
docs["security-handbook.md"] = security
for index, (heading, (fact, question, quote)) in enumerate(BURIED.items()):
    add_question(
        f"buried-{index}", question, "security-handbook.md", heading, quote, fact, "buried"
    )

# -- distractors: plausible corporate text that shares HR vocabulary ------------------------------

TEAMS = [
    "facilities",
    "brand",
    "procurement",
    "marketing",
    "sustainability",
    "safety",
    "canteen",
    "events",
    "training",
    "library",
    "legal",
    "finance operations",
]
ARTIFACTS = [
    "a policy update",
    "an approval checklist",
    "a quarterly newsletter",
    "a guidance note",
    "a request form",
    "a summary of leave arrangements for contractors",
    "an annual report",
]
GOALS = [
    "understand approvals",
    "plan their days",
    "follow the correct process",
    "find the right contact",
    "meet deadlines",
    "reduce risk",
]
for team in TEAMS:
    body = f"# {team.title()} Guide\n\nThis guide is maintained by the {team} team.\n"
    for section in range(5):
        sentences = [
            f"The {team} team publishes {rng.choice(ARTIFACTS)} every {rng.choice(['month', 'quarter', 'year'])} to help employees {rng.choice(GOALS)}."
            for _ in range(3)
        ]
        sentences.append(
            f"Requests are normally answered within {rng.randint(2, 12)} working days and need manager approval."
        )
        body += (
            f"\n## {rng.choice(['Overview', 'Requests', 'Approvals', 'Schedules', 'Contacts', 'Standards'])} {section + 1}\n\n"
            + " ".join(sentences)
            + "\n"
        )
    docs[f"guide-{team.replace(' ', '-')}.md"] = body

# -- follow-ups -------------------------------------------------------------------------------------

for index, (first, second) in enumerate([(2, 3), (4, 5), (0, 6), (7, 1)]):
    a, b = COUNTRIES[first], COUNTRIES[second]
    facts = {**country_facts[b], "cover_fmt": f"{country_facts[b]['cover']:,}"}
    slug = b.lower().replace(" ", "-")
    history = [
        {"role": "user", "content": f"How many days of annual leave do employees in {a} get?"},
        {
            "role": "assistant",
            "content": f"Employees in {a} get {country_facts[a]['leave']} days of annual leave.",
        },
    ]
    add_question(
        f"followup-{index}",
        f"And in {b}?",
        f"supplement-{slug}.md",
        "Annual leave",
        "{leave} days of paid annual leave".format(**facts),
        f"{facts['leave']} days",
        "followup",
        history,
    )
for index, (first, second) in enumerate(
    [(3, "notice"), (5, "remote"), (1, "overtime_x"), (6, "parental")]
):
    country = COUNTRIES[first]
    slug = country.lower().replace(" ", "-")
    facts = country_facts[country]
    section, ask, quote = {
        "notice": (
            "Notice period",
            "What is the notice period there?",
            f"{facts['notice']} days written notice",
        ),
        "remote": (
            "Remote work",
            "How many remote days per week are allowed there?",
            f"up to {facts['remote']} days per week",
        ),
        "overtime_x": (
            "Overtime",
            "What rate is overtime paid at there?",
            f"paid at {facts['overtime_x']} times the base hourly rate",
        ),
        "parental": (
            "Parental leave",
            "How many weeks of parental leave for primary caregivers there?",
            f"{facts['parental']} weeks of paid parental leave",
        ),
    }[second]
    history = [
        {"role": "user", "content": f"How many paid public holidays does {country} have?"},
        {
            "role": "assistant",
            "content": f"{country} has {facts['holidays']} paid public holidays.",
        },
    ]
    add_question(
        f"followup-{index + 4}",
        ask,
        f"supplement-{slug}.md",
        section,
        quote,
        quote,
        "followup",
        history,
    )

# -- things the documents cannot answer -------------------------------------------------------------

for index, text in enumerate(
    [
        "What is the company's stock option vesting schedule?",
        "How many days of paternity leave are given in Australia?",
        "What is the CEO's bonus this year?",
        "Is there a company car scheme?",
        "How do I reset my email password?",
    ]
):
    questions.append(
        {
            "id": f"none-{index}",
            "question": text,
            "answerable": False,
            "tags": ["unanswerable"],
            "split": "test" if index % 2 else "dev",
        }
    )

# -- write files and verify every gold quote really appears in its document -------------------------


def normalise(text: str) -> str:
    return re.sub(r"[\s*_`|#>]+", " ", text).lower().strip()


docs_dir = HERE / "docs"
docs_dir.mkdir(exist_ok=True)
for existing in docs_dir.glob("*.md"):
    existing.unlink()
for name, text in docs.items():
    (docs_dir / name).write_text(text, encoding="utf-8")

problems = []
for case in questions:
    for ref in case.get("gold_sources", []):
        if normalise(ref["quote"]) not in normalise(docs[ref["source"]]):
            problems.append(f"{case['id']}: quote not found in {ref['source']}: {ref['quote']!r}")
if problems:
    raise SystemExit("\n".join(problems))
(HERE / "questions.jsonl").write_text(
    "\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8"
)

by_tag: dict[str, int] = {}
for case in questions:
    by_tag[case["tags"][0]] = by_tag.get(case["tags"][0], 0) + 1
print(f"{len(docs)} documents, {len(questions)} questions: {by_tag}")
