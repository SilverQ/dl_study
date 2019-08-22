"""
Directions Reduction
Instructions
    Once upon a time, on a way through the old wild west,…
    … a man was given directions to go from one point to another. The directions were "NORTH", "SOUTH", "WEST", "EAST".
    Clearly "NORTH" and "SOUTH" are opposite, "WEST" and "EAST" too.
    Going to one direction and coming back the opposite direction is a needless effort.
    Since this is the wild west, with dreadful weather and not much water,
    it's important to save yourself some energy, otherwise you might die of thirst!
    How I crossed the desert the smart way.
    The directions given to the man are, for example, the following (depending on the language):
    ["NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST"].
    or
    { "NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST" };
    or
    [North, South, South, East, West, North, West]
    You can immediately see that going "NORTH" and then "SOUTH" is not reasonable, better stay to the same place!
    So the task is to give to the man a simplified version of the plan. A better plan in this case is simply:
    ["WEST"]
    or
    { "WEST" }
    or
    [West]
    Other examples:
    In ["NORTH", "SOUTH", "EAST", "WEST"], the direction "NORTH" + "SOUTH" is going north and coming back right away.
    What a waste of time! Better to do nothing.
    The path becomes ["EAST", "WEST"], now "EAST" and "WEST" annihilate each other, therefore, the final result is []
    In ["NORTH", "EAST", "WEST", "SOUTH", "WEST", "WEST"], "NORTH" and "SOUTH" are not directly opposite
    but they become directly opposite after the reduction of "EAST" and "WEST"
    so the whole path is reducible to ["WEST", "WEST"].

    Task
    Write a function dirReduc which will take an array of strings and returns an array of strings
    with the needless directions removed (W<->E or S<->N side by side).
        The Haskell version takes a list of directions with data Direction = North | East | West | South.
        The Clojure version returns nil when the path is reduced to nothing.
        The Rust version takes a slice of enum Direction {NORTH, SOUTH, EAST, WEST}.
    See more examples in "Sample Tests:"
    Notes
        Not all paths can be made simpler. The path ["NORTH", "WEST", "SOUTH", "EAST"] is not reducible.
        "NORTH" and "WEST", "WEST" and "SOUTH", "SOUTH" and "EAST" are not directly opposite of each other
        and can't become such. Hence the result path is itself : ["NORTH", "WEST", "SOUTH", "EAST"].
        if you want to translate, please ask before translating.
"""


def opposite(dir):
    direction = {"NORTH": 1, "SOUTH": 4, "EAST": 2, "WEST": 3}
    # print(dir, check == 5)
    return True if sum(direction[i.upper()] for i in dir) == 5 else False


def scan(arr):
    return [i for i in range(0, len(arr)-1) if len(arr[i:i+2]) == 2 and opposite(arr[i:i+2])]


def dirReduc(arr):
    flag = True
    while flag:
        print('input: ', arr, '\n', 'scan_result: ', scan(arr))
        if len(scan(arr)) == 0:
            flag = False
        else:
            print('delete: ', scan(arr)[0], "'th elements : ", arr[scan(arr)[0]:scan(arr)[0]+2])
            del arr[scan(arr)[0]:scan(arr)[0]+2]
        # for i in scan(arr)[::-1]:
        #     # print(arr[i:i+2])
        #     del arr[i:i + 2]
    return arr
    # 끝까지 스캔했을 때 더이상 삭제할 조합이 없으면 종료
    # 처음 입력된 길이만큼 스캔, True가 발생하면 위치를 기록해두고 스캔 완료
    # True 발생 위치, 다음 객체를 삭제하고 반복
    # 더 이상 True가 발생하지 않으면 반복 종료

    # print(opposite("NORTH", "SOUTH"))
    # print(opposite("EAST", "WEST"))
    # print(opposite("NORTH", "EAST"))


# a = ["NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST"]
# test.assert_equals(dirReduc(a), ['WEST'])
# u=["NORTH", "WEST", "SOUTH", "EAST"]
# test.assert_equals(dirReduc(u), ["NORTH", "WEST", "SOUTH", "EAST"])


# a = ["NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "EAST", "NORTH", "SOUTH"]
# print(dirReduc(a))  # ['WEST']
u = ["NORTH", "WEST", "SOUTH", "EAST"]
print(dirReduc(u))  # ["NORTH", "WEST", "SOUTH", "EAST"]
# print(dirReduc(['EAST', 'NORTH', 'EAST', 'WEST']))
# print(dirReduc(['NORTH', 'NORTH', 'EAST', 'SOUTH', 'EAST', 'EAST', 'SOUTH', 'SOUTH', 'SOUTH', 'NORTH']))
# ['NORTH', 'NORTH', 'EAST', 'SOUTH', 'EAST', 'EAST', 'SOUTH', 'SOUTH']


# Test Results:
# Test Passed
# ['NORTH'] should equal ['NORTH', 'NORTH']
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed
# Test Passed


# Other's Answers


# Unnamed, aenik97, CSNqwer, kickh, joei26, suhlob (plus 4 more warriors)
opposite = {'NORTH': 'SOUTH', 'EAST': 'WEST', 'SOUTH': 'NORTH', 'WEST': 'EAST'}
def dirReduc(plan):
    new_plan = []
    for d in plan:
        if new_plan and new_plan[-1] == opposite[d]:
            new_plan.pop()
        else:
            new_plan.append(d)
    return new_plan


# Chrisi, BobrovAE308, Mario Montalvo, pitchet95, zhi_wang, Self Lee (plus 4 more warriors)
def dirReduc(arr):
    dir = " ".join(arr)
    dir2 = dir.replace("NORTH SOUTH",'').replace("SOUTH NORTH",'').replace("EAST WEST",'').replace("WEST EAST",'')
    dir3 = dir2.split()
    return dirReduc(dir3) if len(dir3) < len(arr) else dir3


# 766dt
def dirReduc(arr):
    opposites = [{'NORTH', 'SOUTH'}, {'EAST', 'WEST'}]

    for i in range(len(arr) - 1):
        if set(arr[i:i + 2]) in opposites:
            del arr[i:i + 2]
            return dirReduc(arr)

    return arr
