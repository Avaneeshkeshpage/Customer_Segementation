"""
marketing_actions.py
-----------------
Maps each customer segment/persona to a recommended marketing action.
This is the piece that turns "which cluster is this customer in" into
something a marketing team can actually act on.
"""

MARKETING_ACTIONS = {
    "Champions": {
        "description": "Recent, frequent, high-spending customers.",
        "action": "Reward with loyalty perks, early access to new products, and referral incentives.",
    },
    "Loyal Customers": {
        "description": "Consistently active and reliable spenders.",
        "action": "Upsell premium products and enroll them in a loyalty/rewards program.",
    },
    "Potential Loyalists": {
        "description": "Recently active with growing engagement, but not yet high-frequency.",
        "action": "Encourage repeat purchases with limited-time offers and personalized recommendations.",
    },
    "At Risk": {
        "description": "Used to purchase regularly, but haven't been active recently.",
        "action": "Send a win-back campaign with a personalized discount or 'we miss you' offer.",
    },
    "New Customers": {
        "description": "Recently joined, with limited purchase history so far.",
        "action": "Send onboarding content and a first-purchase incentive to encourage a second order.",
    },
    "Low-Value / Lapsed": {
        "description": "Infrequent, low-spend, and largely inactive customers.",
        "action": "Include in low-cost broad campaigns only; avoid high-cost targeted spend.",
    },
}


# Separate from MARKETING_ACTIONS above: these apply to customers who don't
# yet have enough transaction history for their K-Means segment to be
# trustworthy (see cold_start.py). They use a fixed rule-based action instead
# of a behavioral cluster assignment.
NEW_CUSTOMER_ACTIONS = {
    "no_history": {
        "description": "Signed up but hasn't made a purchase yet.",
        "action": "Send a welcome email/notification with a first-purchase discount to drive the initial order.",
    },
    "cold_start": {
        "description": "Made 1 purchase very recently — too little history for reliable segmentation.",
        "action": "Send a second-purchase nudge (e.g. a small loyalty invite or complementary product suggestion) to move them toward becoming a repeat customer.",
    },
}


def get_new_customer_action(status: str) -> dict:
    return NEW_CUSTOMER_ACTIONS.get(
        status,
        {"description": "Unknown status.", "action": "Review manually."}
    )


def get_action(segment_name: str) -> dict:
    return MARKETING_ACTIONS.get(
        segment_name,
        {"description": "Unclassified segment.", "action": "Review manually."}
    )
