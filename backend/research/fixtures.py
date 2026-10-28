"""Three offline B2B research examples. Deterministic text, not model output.

Every sentence here is authored fixture copy. Nothing in this module is a
customer interview, a market measurement, or a provider response.
"""
from __future__ import annotations

from typing import Any


def _validation(vid: str, text: str) -> dict[str, str]:
    return {"id": vid, "kind": "human_validation_need", "text": text}


def _statement(
    sid: str,
    participant_id: str,
    text: str,
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": sid,
        "kind": "synthetic_statement",
        "participant_id": participant_id,
        "text": text,
        "source_refs": list(source_refs or []),
    }


def _finding(
    fid: str,
    title: str,
    text: str,
    source_ids: list[str],
    validation_id: str,
    validation_text: str,
) -> dict[str, Any]:
    return {
        "id": fid,
        "kind": "interpretation",
        "title": title,
        "text": text,
        "source_ids": source_ids,
        "human_validation": _validation(validation_id, validation_text),
    }


def _person(
    pid: str,
    name: str,
    role: str,
    perspective: str,
) -> dict[str, Any]:
    return {
        "id": pid,
        "kind": "synthetic_participant",
        "name": name,
        "role": role,
        "perspective": perspective,
        "synthetic": True,
        "human_subject": False,
    }


SAAS_PRICING: dict[str, Any] = {
    "slug": "saas-pricing",
    "project": {
        "id": "prj_saas_pricing",
        "name": "Northline Analytics",
        "organization": "Fictional B2B vendor — not a client engagement",
    },
    "study": {
        "id": "stu_saas_pricing",
        "title": "Seat price to usage price",
        "example_slug": "saas-pricing",
        "status": "findings_ready",
        "requested_participant_count": 4,
        "research_question": {
            "id": "q_saas_pricing",
            "kind": "user_input",
            "text": (
                "How do finance, revenue operations, IT, and a team lead at a "
                "mid-market analytics vendor describe a move from named-seat pricing "
                "to a usage component on an analytics add-on?"
            ),
        },
        "context": {
            "id": "ctx_saas_pricing",
            "kind": "user_input",
            "text": (
                "User-supplied product brief: the vendor is considering a usage "
                "component for an analytics add-on. Buyers currently purchase named "
                "seats on an annual term. No customer interviews have been run. "
                "This brief is context typed by a person, not a market measurement."
            ),
            "sources": [
                {
                    "id": "src_saas_pricing_page",
                    "kind": "source",
                    "title": "User-supplied pricing note",
                    "supplied_by": "user",
                    "text": (
                        "Note supplied by the user: the current public offer lists "
                        "named seats and an annual contract. The note is a document, "
                        "not a finding about willingness to pay."
                    ),
                },
                {
                    "id": "src_saas_support_note",
                    "kind": "source",
                    "title": "User-supplied support theme",
                    "supplied_by": "user",
                    "text": (
                        "Note supplied by the user: some tickets mention surprise "
                        "invoices and unclear seat counts. The note is an anecdote "
                        "from the user, not a counted ticket sample."
                    ),
                },
            ],
        },
        "audience": {
            "id": "aud_saas_pricing",
            "name": "Analytics buying circle",
            "description": (
                "Four synthetic roles around a seat-to-usage change. "
                "An authored roster, not a market sample."
            ),
            "synthetic": True,
            "representativeness": None,
        },
        "participants": [
            _person(
                "par_saas_mara",
                "Mara Ellison",
                "Finance director",
                "Reads contract changes through budget signature and invoice control.",
            ),
            _person(
                "par_saas_jonas",
                "Jonas Keller",
                "Revenue operations manager",
                "Has to explain the price page to an existing seat buyer.",
            ),
            _person(
                "par_saas_priya",
                "Priya Nair",
                "IT administrator",
                "Wants a shutoff and a named data boundary before a meter is on.",
            ),
            _person(
                "par_saas_elena",
                "Elena Vogt",
                "Analytics team lead",
                "Watches whether a meter makes people avoid the add-on.",
            ),
        ],
        "sessions": [
            {
                "id": "ses_saas_pricing",
                "kind": "research_session",
                "execution": "offline_deterministic_fixture",
                "genuine_ai": False,
                "provider": None,
                "provider_calls": 0,
                "cost_usd": 0,
                "rounds": 1,
                "live_llm": False,
                "label": "Offline deterministic fixture — not genuine AI research",
                "statements": [
                    _statement(
                        "stmt_saas_mara_1",
                        "par_saas_mara",
                        "Mara Ellison is a synthetic finance director, not a human subject. "
                        "In this offline fixture she treats a usage line that can change "
                        "the invoice after signature as a procurement risk. This is not "
                        "an interview quote and not a frequency.",
                    ),
                    _statement(
                        "stmt_saas_mara_2",
                        "par_saas_mara",
                        "Mara Ellison wants the record to keep the user-supplied pricing "
                        "note in view: named seats and an annual term are what that note "
                        "lists. She does not convert the note into a willingness-to-pay figure.",
                        ["src_saas_pricing_page"],
                    ),
                    _statement(
                        "stmt_saas_jonas_1",
                        "par_saas_jonas",
                        "Jonas Keller is a synthetic revenue-operations persona. He says a "
                        "usage component is hard to explain to an existing seat buyer unless "
                        "the contract still shows the seat line during a transition. Not an interview.",
                    ),
                    _statement(
                        "stmt_saas_jonas_2",
                        "par_saas_jonas",
                        "Jonas Keller points at the user-supplied support note about surprise "
                        "invoices and unclear seat counts. He treats those words as the user's "
                        "anecdote, not as a counted ticket sample.",
                        ["src_saas_support_note"],
                    ),
                    _statement(
                        "stmt_saas_priya_1",
                        "par_saas_priya",
                        "Priya Nair is a synthetic IT administrator. She withholds a usage "
                        "meter until an admin can see who consumed the add-on and can turn "
                        "a seat off. This is a fixture statement.",
                    ),
                    _statement(
                        "stmt_saas_priya_2",
                        "par_saas_priya",
                        "Priya Nair adds that the user-supplied brief does not name a data "
                        "boundary, so she refuses to invent one. A person has to supply that document.",
                        ["ctx_saas_pricing"],
                    ),
                    _statement(
                        "stmt_saas_elena_1",
                        "par_saas_elena",
                        "Elena Vogt is a synthetic team lead. She worries people will avoid "
                        "the analytics add-on if a usage meter feels like surveillance of "
                        "their curiosity. Not a measured adoption rate.",
                    ),
                    _statement(
                        "stmt_saas_elena_2",
                        "par_saas_elena",
                        "Elena Vogt wants a pilot her team can stop without a surprise invoice. "
                        "She cites no figure because the brief does not contain one.",
                        ["ctx_saas_pricing"],
                    ),
                ],
            }
        ],
        "findings": [
            _finding(
                "find_saas_procurement",
                "Procurement risk in the finance persona",
                "Interpretation only: the synthetic finance director treats an invoice "
                "that can move after signature as a procurement risk, set against the "
                "user-supplied pricing note. This is not evidence that finance buyers "
                "block usage pricing.",
                ["stmt_saas_mara_1", "src_saas_pricing_page"],
                "hv_saas_procurement",
                "A person must check real procurement reviews before anyone treats "
                "invoice movement as a buying blocker. Statement stmt_saas_mara_1 is synthetic.",
            ),
            _finding(
                "find_saas_transition",
                "Seat line as a transition bridge in this fixture",
                "Interpretation only: the synthetic revenue-operations persona wants the "
                "seat line to remain visible during a transition, and treats the support "
                "note as a user anecdote about surprise invoices. No ticket volume is claimed.",
                ["stmt_saas_jonas_1", "stmt_saas_jonas_2", "src_saas_support_note"],
                "hv_saas_transition",
                "A person must read the actual tickets before the support note is retold "
                "as a theme. The persona is not a customer.",
            ),
            _finding(
                "find_saas_admin",
                "Admin shutoff before a meter",
                "Interpretation only: the synthetic IT persona withholds a usage meter "
                "until an admin can see consumption and turn a seat off. The brief does "
                "not provide a data boundary, and none is invented here.",
                ["stmt_saas_priya_1", "ctx_saas_pricing"],
                "hv_saas_admin",
                "A person must supply the real access and data note. Do not publish the "
                "synthetic shutoff line as an IT requirement from research.",
            ),
            _finding(
                "find_saas_avoidance",
                "Avoidance worry from the team-lead persona",
                "Interpretation only: the synthetic team lead voices a worry that a usage "
                "meter could make people avoid the add-on. That worry is not an adoption metric.",
                ["stmt_saas_elena_1", "q_saas_pricing"],
                "hv_saas_avoidance",
                "A person must talk with people who use the add-on before this worry is "
                "treated as behavior. The sentence is a fixture.",
            ),
        ],
    },
}


BANKING_ONBOARDING: dict[str, Any] = {
    "slug": "banking-onboarding",
    "project": {
        "id": "prj_banking_onboarding",
        "name": "Harbor Mutual",
        "organization": "Fictional institution — not a client engagement",
    },
    "study": {
        "id": "stu_banking_onboarding",
        "title": "Retail checking onboarding",
        "example_slug": "banking-onboarding",
        "status": "findings_ready",
        "requested_participant_count": 5,
        "research_question": {
            "id": "q_banking_onboarding",
            "kind": "user_input",
            "text": (
                "What frictions do branch, KYC, digital product, support, and "
                "compliance personas describe when a new retail customer opens a "
                "checking account?"
            ),
        },
        "context": {
            "id": "ctx_banking_onboarding",
            "kind": "user_input",
            "text": (
                "User-supplied process note: a fictional mutual lets a person start "
                "a checking application online and finish identity checks in a branch. "
                "No applicant study has been run. Staff quotes below are synthetic."
            ),
            "sources": [
                {
                    "id": "src_banking_process",
                    "kind": "source",
                    "title": "User-supplied onboarding steps",
                    "supplied_by": "user",
                    "text": (
                        "Steps supplied by the user: identity document, address, "
                        "source-of-funds line, then a branch visit if the online check "
                        "stops. The list is a process sketch, not a count of stopped applications."
                    ),
                },
                {
                    "id": "src_banking_script",
                    "kind": "source",
                    "title": "User-supplied branch script excerpt",
                    "supplied_by": "user",
                    "text": (
                        "Excerpt supplied by the user: the branch script tells the advisor "
                        "to ask where deposited money comes from when the line is blank. "
                        "The excerpt is a document, not evidence of applicant behavior."
                    ),
                },
            ],
        },
        "audience": {
            "id": "aud_banking_onboarding",
            "name": "Checking onboarding circle",
            "description": (
                "Five synthetic staff roles around one fictional checking journey. "
                "Not account holders and not a market sample."
            ),
            "synthetic": True,
            "representativeness": None,
        },
        "participants": [
            _person(
                "par_bank_helen",
                "Helen Okonkwo",
                "Branch advisor",
                "Sits with the applicant when the online check stops.",
            ),
            _person(
                "par_bank_farid",
                "Farid Mansour",
                "KYC analyst",
                "Reads the identity packet after the branch visit.",
            ),
            _person(
                "par_bank_lina",
                "Lina Berg",
                "Digital product owner",
                "Owns the online form the applicant sees first.",
            ),
            _person(
                "par_bank_samuel",
                "Samuel Ortiz",
                "Support lead",
                "Hears applicants who do not know why the application stopped.",
            ),
            _person(
                "par_bank_nora",
                "Nora Iversen",
                "Compliance reviewer",
                "Stops the file when source of funds is blank.",
            ),
        ],
        "sessions": [
            {
                "id": "ses_banking_onboarding",
                "kind": "research_session",
                "execution": "offline_deterministic_fixture",
                "genuine_ai": False,
                "provider": None,
                "provider_calls": 0,
                "cost_usd": 0,
                "rounds": 1,
                "live_llm": False,
                "label": "Offline deterministic fixture — not genuine AI research",
                "statements": [
                    _statement(
                        "stmt_bank_helen_1",
                        "par_bank_helen",
                        "Helen Okonkwo is a synthetic branch advisor, not a staff interviewee. "
                        "She says the visit goes quiet when she has to re-ask for an identity "
                        "document the online form already requested. Not a count of visits.",
                        ["src_banking_process"],
                    ),
                    _statement(
                        "stmt_bank_helen_2",
                        "par_bank_helen",
                        "Helen Okonkwo follows the user-supplied script and asks where money "
                        "comes from when that line is blank. She does not invent how often the line is blank.",
                        ["src_banking_script"],
                    ),
                    _statement(
                        "stmt_bank_farid_1",
                        "par_bank_farid",
                        "Farid Mansour is a synthetic KYC analyst. He says a photo of an identity "
                        "document he cannot read sends the file back, and he will not guess what "
                        "share of files do that.",
                    ),
                    _statement(
                        "stmt_bank_farid_2",
                        "par_bank_farid",
                        "Farid Mansour wants the branch note and the online packet in one view. "
                        "The user-supplied step list keeps them as separate steps, and he does not merge them into a metric.",
                        ["src_banking_process"],
                    ),
                    _statement(
                        "stmt_bank_lina_1",
                        "par_bank_lina",
                        "Lina Berg is a synthetic digital product owner. She says the online form "
                        "should show why a check stopped, in plain language, before it offers a branch time.",
                    ),
                    _statement(
                        "stmt_bank_lina_2",
                        "par_bank_lina",
                        "Lina Berg refuses to add a completion figure. The user-supplied note has "
                        "steps only, and a missing number stays missing.",
                        ["ctx_banking_onboarding"],
                    ),
                    _statement(
                        "stmt_bank_samuel_1",
                        "par_bank_samuel",
                        "Samuel Ortiz is a synthetic support lead. He describes callers who know "
                        "the application stopped and do not know which line to fix. He names no volume.",
                    ),
                    _statement(
                        "stmt_bank_samuel_2",
                        "par_bank_samuel",
                        "Samuel Ortiz wants the same reason text the product owner asked for, "
                        "so support does not paraphrase a rule they cannot see.",
                        ["stmt_bank_lina_1"],
                    ),
                    _statement(
                        "stmt_bank_nora_1",
                        "par_bank_nora",
                        "Nora Iversen is a synthetic compliance reviewer. She stops the file when "
                        "the source-of-funds line is blank. She will not turn that stop into a rate of failed applications.",
                        ["src_banking_script"],
                    ),
                    _statement(
                        "stmt_bank_nora_2",
                        "par_bank_nora",
                        "Nora Iversen says a friendlier form does not replace the blank-line stop. "
                        "The script excerpt is the document she points at, not a finding about applicants.",
                        ["src_banking_script"],
                    ),
                ],
            }
        ],
        "findings": [
            _finding(
                "find_bank_reread",
                "Repeated identity ask in the branch persona",
                "Interpretation only: the synthetic branch advisor describes a quiet visit "
                "when she re-asks for a document the online step already requested. "
                "No visit volume is claimed.",
                ["stmt_bank_helen_1", "src_banking_process"],
                "hv_bank_reread",
                "A person must watch a real branch conversation or read a real procedure "
                "before this re-ask is treated as a friction customers hit.",
            ),
            _finding(
                "find_bank_blank_line",
                "Blank source-of-funds line stops the file",
                "Interpretation only: the synthetic compliance reviewer stops a file when "
                "source of funds is blank, matching the user-supplied script excerpt. "
                "That stop is not a failure rate.",
                ["stmt_bank_nora_1", "src_banking_script"],
                "hv_bank_blank_line",
                "A compliance officer must confirm the real rule. Do not quote Nora Iversen "
                "as a human reviewer.",
            ),
            _finding(
                "find_bank_reason_text",
                "Shared reason text across product and support",
                "Interpretation only: the synthetic product owner and the synthetic support "
                "lead both ask for the same plain-language reason when a check stops. "
                "Agreement here is between fixtures, not between staff studies.",
                ["stmt_bank_lina_1", "stmt_bank_samuel_2"],
                "hv_bank_reason_text",
                "A person must compare the live form and the support macro. The two "
                "personas did not interview each other.",
            ),
            _finding(
                "find_bank_unreadable",
                "Unreadable identity packet returns the file",
                "Interpretation only: the synthetic KYC analyst sends back a packet he "
                "cannot read and refuses to estimate how often that happens. "
                "Keep that refusal; do not fill the gap with a number.",
                ["stmt_bank_farid_1", "src_banking_process"],
                "hv_bank_unreadable",
                "A person must review real packets. The analyst sentence is synthetic "
                "and is not evidence of a document-quality problem in the market.",
            ),
        ],
    },
}


RESTAURANT_SOFTWARE: dict[str, Any] = {
    "slug": "restaurant-software",
    "project": {
        "id": "prj_restaurant_software",
        "name": "Tableline",
        "organization": "Fictional software vendor — not a client engagement",
    },
    "study": {
        "id": "stu_restaurant_software",
        "title": "POS and reservations in one dining room",
        "example_slug": "restaurant-software",
        "status": "findings_ready",
        "requested_participant_count": 3,
        "research_question": {
            "id": "q_restaurant_software",
            "kind": "user_input",
            "text": (
                "How do an owner, a floor manager, and a server persona talk about "
                "adopting one tool that combines the point of sale and reservations "
                "for an independent restaurant?"
            ),
        },
        "context": {
            "id": "ctx_restaurant_software",
            "kind": "user_input",
            "text": (
                "User-supplied vendor brief: Tableline wants to replace a separate "
                "reservation book and a separate point-of-sale tablet in one dining "
                "room. No operator interviews have been run. Payback is not in the brief "
                "and must not be invented."
            ),
            "sources": [
                {
                    "id": "src_restaurant_brief",
                    "kind": "source",
                    "title": "User-supplied vendor brief",
                    "supplied_by": "user",
                    "text": (
                        "Brief supplied by the user: one dining room, a paper reservation "
                        "book, and a point-of-sale tablet that does not see the book. "
                        "The brief is a scene, not a census of restaurants."
                    ),
                },
                {
                    "id": "src_restaurant_shift",
                    "kind": "source",
                    "title": "User-supplied shift note",
                    "supplied_by": "user",
                    "text": (
                        "Note supplied by the user: during service the floor manager "
                        "calls out covers from the book while servers enter items on the "
                        "tablet. The note is one scene from the user, not a timed study."
                    ),
                },
            ],
        },
        "audience": {
            "id": "aud_restaurant_software",
            "name": "Independent dining-room circle",
            "description": (
                "Three synthetic roles in one fictional dining room. "
                "Not a panel of operators and not a market sample."
            ),
            "synthetic": True,
            "representativeness": None,
        },
        "participants": [
            _person(
                "par_rest_gio",
                "Gio Rossi",
                "Owner",
                "Decides whether a second tablet is worth the training week.",
            ),
            _person(
                "par_rest_amira",
                "Amira Haddad",
                "Floor manager",
                "Runs the book and the pass during service.",
            ),
            _person(
                "par_rest_ben",
                "Ben Cho",
                "Server",
                "Enters items while guests are still deciding.",
            ),
        ],
        "sessions": [
            {
                "id": "ses_restaurant_software",
                "kind": "research_session",
                "execution": "offline_deterministic_fixture",
                "genuine_ai": False,
                "provider": None,
                "provider_calls": 0,
                "cost_usd": 0,
                "rounds": 1,
                "live_llm": False,
                "label": "Offline deterministic fixture — not genuine AI research",
                "statements": [
                    _statement(
                        "stmt_rest_gio_1",
                        "par_rest_gio",
                        "Gio Rossi is a synthetic owner, not an interviewed operator. "
                        "He will not guess a payback period. The brief does not contain one, "
                        "and he will not invent one.",
                        ["ctx_restaurant_software"],
                    ),
                    _statement(
                        "stmt_rest_gio_2",
                        "par_rest_gio",
                        "Gio Rossi says he would try the combined tool only if the paper book "
                        "can stay at the stand for the first service week. He points at the "
                        "user-supplied scene of one dining room, not at other restaurants.",
                        ["src_restaurant_brief"],
                    ),
                    _statement(
                        "stmt_rest_amira_1",
                        "par_rest_amira",
                        "Amira Haddad is a synthetic floor manager. She says calling covers "
                        "from the book while servers use a separate tablet is the friction "
                        "in the user-supplied shift note. She does not time that friction.",
                        ["src_restaurant_shift"],
                    ),
                    _statement(
                        "stmt_rest_amira_2",
                        "par_rest_amira",
                        "Amira Haddad wants a reservation to show on the same screen as the "
                        "check so she stops repeating the table number. That want is a fixture, "
                        "not a feature-priority score.",
                    ),
                    _statement(
                        "stmt_rest_ben_1",
                        "par_rest_ben",
                        "Ben Cho is a synthetic server. He says a new screen during the first "
                        "greeting makes him look down while the guest is still talking. "
                        "Not a measured service-time change.",
                    ),
                    _statement(
                        "stmt_rest_ben_2",
                        "par_rest_ben",
                        "Ben Cho will keep the old tablet path until the combined tool can "
                        "take the same items he enters today. He refuses to call that a "
                        "preference share.",
                        ["src_restaurant_shift"],
                    ),
                ],
            }
        ],
        "findings": [
            _finding(
                "find_rest_no_payback",
                "No payback figure is available",
                "Interpretation only: the synthetic owner refuses to state a payback period "
                "because the user-supplied brief has none. The gap stays open. "
                "Do not fill it.",
                ["stmt_rest_gio_1", "ctx_restaurant_software"],
                "hv_rest_no_payback",
                "A person must bring real cost notes if a payback discussion is required. "
                "The owner sentence is synthetic and is not financial research.",
            ),
            _finding(
                "find_rest_book_bridge",
                "Paper book held for a first service week",
                "Interpretation only: the synthetic owner and the user-supplied scene keep "
                "the paper book in reach during an early service. This is a fixture condition, "
                "not evidence that operators require a parallel book.",
                ["stmt_rest_gio_2", "src_restaurant_brief"],
                "hv_rest_book_bridge",
                "A person must ask operators who have switched tools. Do not treat Gio Rossi "
                "as a human owner.",
            ),
            _finding(
                "find_rest_one_screen",
                "One screen for the book and the check",
                "Interpretation only: the synthetic floor manager wants the reservation and "
                "the check on one screen, and the synthetic server wants the old item path "
                "to keep working. Neither sentence is a priority ranking.",
                ["stmt_rest_amira_2", "stmt_rest_ben_2", "src_restaurant_shift"],
                "hv_rest_one_screen",
                "A person must watch a real service before these two wants are written as "
                "product requirements. Both speakers are synthetic.",
            ),
        ],
    },
}


EXAMPLES: dict[str, dict[str, Any]] = {
    "saas-pricing": SAAS_PRICING,
    "banking-onboarding": BANKING_ONBOARDING,
    "restaurant-software": RESTAURANT_SOFTWARE,
}

EXAMPLE_ORDER = ("saas-pricing", "banking-onboarding", "restaurant-software")


def get_example(slug: str) -> dict[str, Any] | None:
    spec = EXAMPLES.get(slug)
    if spec is None:
        return None
    # Deep copy via json would drop nothing; callers must copy before mutate.
    import copy

    return copy.deepcopy(spec)
