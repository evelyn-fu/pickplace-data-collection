from enum import Enum

class PlannerState(Enum):
    # Two grasp
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    START = 2
    PLAN_PICK = 3
    GO_TO_PICK_PREGRASP = 4
    PICK_GRASP = 5
    GO_HOME0 = 6
    SCANNING1 = 7
    GO_TO_PREGRASP1 = 8
    GRASP1 = 9
    GO_HOME1 = 10
    SCANNING2 = 11
    GO_TO_PREGRASP2 = 12
    GRASP2 = 13
    GO_HOME2 = 14
    PLAN_SYS_ID = 15
    GO_TO_SYS_ID_PREGRASP = 16
    SYS_ID_GRASP = 17
    GO_HOME3 = 18
    RESET = 19
    DONE = 20

    # Turntable
    GO_TO_SPINNING = 21
    SPINNING = 22

    # Additional Regrasps
    SCANNINGN = 23
    GO_TO_PREGRASPN = 24
    GRASPN = 25
    GO_HOMEN = 26
    
class PickState(Enum):
    IDLE = 1
    PICK = 2
    TO_DISPLAY = 3
    DISPLAY = 4
    TO_PLACE = 5
    PLACE = 6