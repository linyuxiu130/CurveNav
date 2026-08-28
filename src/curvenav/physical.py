"""Physical contract of the benchmark Dingo in base-link coordinates."""


# These values are measured from the enabled collision prims in the benchmark
# dingo.usd.  The asset is identical in NavDP, X-NavDP and the fixed benchmark.
DINGO_USD_SHA256 = (
    "43db9c54066d833e0bfc91d8d33eb1ff3345d4c8a3cd29f914649bafffbcce20"
)
DINGO_WHEEL_RADIUS_M = 0.06125
DINGO_WHEEL_BASE_M = 0.22616
DINGO_CAMERA_FORWARD_OFFSET_M = 0.28618
DINGO_CAMERA_HEIGHT_M = 0.62532
DINGO_CAMERA_DOWNWARD_PITCH_DEGREES = 10.0

# Exact circular configuration-space envelope of all moving collision shapes.
ROBOT_FOOTPRINT_RADIUS_M = 0.167584539
ROBOT_COLLISION_BOTTOM_Z_M = -0.044000001
ROBOT_COLLISION_TOP_Z_M = 0.117981499
ROBOT_COLLISION_HEIGHT_M = (
    ROBOT_COLLISION_TOP_Z_M - ROBOT_COLLISION_BOTTOM_Z_M
)

# A five-centimetre step is below the 6.125 cm wheel radius.  Habitat uses the
# height from the ground plane; depth points use base-link z and therefore need
# the translated lower bound below.
MAXIMUM_TRAVERSABLE_HEIGHT_M = 0.05
BODY_OBSTACLE_MIN_Z_M = (
    ROBOT_COLLISION_BOTTOM_Z_M + MAXIMUM_TRAVERSABLE_HEIGHT_M
)
EXTRA_CLEARANCE_M = 0.10
