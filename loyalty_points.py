"""
loyalty_points.py
-----------------
A simple points-based loyalty system layered on top of the segmentation.

Rule: every purchase earns points based on amount spent, with a multiplier
that depends on the customer's current segment — this directly reinforces
the "reward Champions" / "upsell Loyal Customers" marketing actions with an
actual mechanic, instead of just a text recommendation.

Points formula:  points = (amount / 100) * segment_multiplier
(i.e. base rate of 1 point per ₹100 spent)
"""

POINTS_PER_RUPEE = 1 / 100  # 1 point per ₹100 spent

# Higher-value segments earn a bonus multiplier — this is what "reward with
# loyalty perks" (Champions) and "upsell + loyalty program" (Loyal Customers)
# from marketing_actions.py actually translates to in points terms.
SEGMENT_MULTIPLIERS = {
    "Champions": 2.0,
    "Loyal Customers": 1.5,
    "Potential Loyalists": 1.25,
    "At Risk": 1.0,
    "New Customers": 1.0,
    "Low-Value / Lapsed": 1.0,
}

# Tier thresholds, checked from highest to lowest. Calibrated against this
# project's sample data so the tiers actually spread customers meaningfully
# (roughly: top ~10% Platinum, next ~15% Gold, next ~25% Silver, rest Bronze).
# If you swap in your own dataset, re-check these against your points
# distribution and adjust — see the "Recalibrating tiers" note in the README.
TIERS = [
    (500, "Platinum"),
    (150, "Gold"),
    (40, "Silver"),
    (0, "Bronze"),
]


def calculate_points(amount: float, segment: str) -> float:
    """Points earned for a single purchase of `amount`, given the customer's segment."""
    multiplier = SEGMENT_MULTIPLIERS.get(segment, 1.0)
    return round(amount * POINTS_PER_RUPEE * multiplier, 1)


def get_tier(total_points: float) -> str:
    """Maps a total points balance to a loyalty tier name."""
    for threshold, tier_name in TIERS:
        if total_points >= threshold:
            return tier_name
    return "Bronze"


def total_points_for_customer(total_monetary: float, segment: str) -> float:
    """
    Approximates a customer's lifetime points using their total historical
    spend and CURRENT segment multiplier.

    Note: this is a simplification — in a real system, points would be
    awarded per transaction using the segment/multiplier active AT THE TIME
    of that purchase, tracked in a running ledger. Since this project only
    stores aggregated RFM totals (not a full transaction-level points
    ledger), we approximate lifetime points using the current segment.
    """
    return calculate_points(total_monetary, segment)
