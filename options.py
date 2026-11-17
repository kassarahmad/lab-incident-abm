"""What a responder does when no reflex applies.

Responders have two reflexes (shoot fire within reach, refill when empty).
Otherwise they keep, for up to `commit` steps, one of three options:

    FOLLOW_TASK   go to the task given by the dispatcher
    NEAREST       go to the nearest fire in my belief map
    EXPLORE       follow the smoke to find a fire I do not know about
"""

FOLLOW_TASK, NEAREST, EXPLORE = range(3)
OPTION_NAMES = ["follow_task", "nearest", "explore"]


def choose_option(valid):
    """Decision rule: a task from the dispatcher first, then the nearest
    fire I know about, otherwise search for one."""
    if FOLLOW_TASK in valid:
        return FOLLOW_TASK
    if NEAREST in valid:
        return NEAREST
    return EXPLORE
