"""Equivalent yaw coordinates for a Euclidean-error legacy control interface."""
import math


def yaw_near_reference(measured, reference):
    """Lift atan2 yaw onto the commanded branch without changing orientation.

    Keep the continuous commanded yaw untouched, including its derivatives.
    Only the measured angle's representation changes by complete turns; this
    avoids artificial 2*pi errors in the retained controller and observer.
    """
    if not math.isfinite(measured) or not math.isfinite(reference):
        raise ValueError('Finite measured and commanded yaw required')
    return reference + math.atan2(math.sin(measured-reference), math.cos(measured-reference))
