# Day 1 Basic Data Type
# 1523. Count Odd Numbers in an Interval Range
class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        return (high +1)/2 -low/2
# Add half the difference of the range.
# Add the endpoints of the range if odd.
# Subtract one if both endpoints are odd.
class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        return (high-low)//2 + high%2 + low%2 - (high%2 and low%2)

# Add half the difference of the range.
# Add one if either endpoint is odd.
class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        return (high-low)//2 + (high%2 or low%2)

# 1491. Average Salary Excluding the Minimum and Maximum Salary
class Solution(object):
    def average(self, salary):
        """
        :type salary: List[int]
        :rtype: float
        """
        return float(sum(salary) - max(salary) - min(salary))/(len(salary)-2) 

# Day 2 Operator
# 191. Number of 1 Bits
# Hamming weight

# Built in solution with Python built-in function:
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        return bin(n).count('1')
# Using bit operation to cancel a 1 in each round
# Think of a number in binary n = XXXXXX1000, n - 1 is XXXXXX0111.
# n & (n - 1) will be XXXXXX0000 which is just remove the last significant 1
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        c = 0
        while n:
            n &= n - 1
            c += 1
        return c

# 1281. Subtract the Product and Sum of Digits of an Integer
class Solution(object):
    def subtractProductAndSum(self, n):
        """
        :type n: int
        :rtype: int
        """
        sum, prod = 0, 1
        while n:
            sum +=  n % 10
            prod *= n % 10
            n //= 10
        return prod - sum

# Day 3 Conditional Statements
# 976. Largest Perimeter Triangle
class Solution(object):
    def largestPerimeter(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        This leads to a simple algorithm: 
        Sort the array. 
        For any cc in the array, we choose the largest possible 
        a <= b <= c these are just the two values adjacent to cc. 
        If this forms a triangle, we return the answer.
        '''
        nums.sort()
        for i in range(len(nums)-3,-1,-1):
            if nums[i] + nums[i+1] > nums[i+2]:
                return nums[i] + nums[i+1] +nums[i+2]
        return 0
class Solution(object):
    def largestPerimeter(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        For a >= b >= c, a,b,c can form a triangle if a < b + c.
        We sort the A
        Try to get a triangle with 3 biggest numbers.
        If A[n-1] < A[n-2] + A[n-3], we get a triangle.
        If A[n-1] >= A[n-2] + A[n-3] >= A[i] + A[j], we cannot get any triangle with A[n-1]
        repeat step2 and step3 with the left numbers.
        '''
        nums = nums.sort(reverse=True) # nums = sorted(nums)[::-1]
        for i in range(len(nums) - 2):
            if nums[i] < nums[i + 1] + nums[i + 2]:
                return nums[i] + nums[i + 1] + nums[i + 2]
        return 0

# 1779. Find Nearest Point That Has the Same X or Y Coordinate
class Solution(object):
    def nearestValidPoint(self, x, y, points):
        """
        :type x: int
        :type y: int
        :type points: List[List[int]]
        :rtype: int
        """
        min_dist, index = float("inf"), -1
        for i, (a, b) in enumerate(points):
            if (a == x or b == y) and abs(a - x) + abs(b - y) < min_dist:
                index, min_dist = i, abs(a - x) + abs(b - y)
        return index

# Day 4 Loop
# 1822. Sign of the Product of an Array
class Solution(object):
    def arraySign(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = 1
        for x in nums: 
            if x == 0: return 0 
            if x < 0: ans *= -1
        return ans 
# 1502. Can Make Arithmetic Progression From Sequence
class Solution(object):
    def canMakeArithmeticProgression(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        mi, mx, n, s = min(arr), max(arr), len(arr), set(arr)
        if (mx - mi) % (n - 1) != 0:
            return False    
        diff = (mx - mi) // (n - 1)
        for _ in range(n):
            if mi not in s:
                return False
            mi += diff
        return True
        
# 202. Happy Number
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        cycle_members = {4, 16, 37, 58, 89, 145, 42, 20}

        def get_next(number):
            total_sum = 0
            while number > 0:
                number, digit = divmod(number, 10)
                total_sum += digit ** 2
            return total_sum

        while n != 1 and n not in cycle_members:
            n = get_next(n)

        return n == 1
# 1790. Check if One String Swap Can Make Strings Equal
class Solution(object):
    def areAlmostEqual(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        diff = [[x, y] for x, y in zip(s1, s2) if x != y]
        return not diff or len(diff) == 2 and diff[0][::-1] == diff[1]
# Day 5 Function
# 589. N-ary Tree Preorder Traversal
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        stack,s=[root],[]
        while stack:
            p=stack.pop()
            if p:
                s.append(p.val)
                for i in range(len(p.children)-1,-1,-1): stack.append(p.children[i])
        return s

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        ret, q = [], root and [root]
        while q:
            node = q.pop()
            ret.append(node.val)
            q += [child for child in node.children[::-1] if child]
        return ret
# 496. Next Greater Element I
'''
Complexity:
Time: O(N + Q), where N <= 1000 is length of nums2 array, Q <= 1000 is length of nums1 array.
Space: O(N + Q)
'''
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        ans = {}
        n = len(nums2)
        st = []
        for i in range(n-1, -1, -1):
            while st and st[-1] <= nums2[i]:
                st.pop()
            if st:
                ans[nums2[i]] = st[-1]
            st.append(nums2[i])
            
        return [ans.get(num, -1) for num in nums1]
        

# 1232. Check If It Is a Straight Line
'''
The slope for a line through any 2 points (x0, y0) and (x1, y1) is (y1 - y0) / (x1 - x0); Therefore, for any given 3 points (denote the 3rd point as (x, y)), if they are in a straight line, the slopes of the lines from the 3rd point to the 2nd point and the 2nd point to the 1st point must be equal:

(y - y1) / (x - x1) = (y1 - y0) / (x1 - x0)
In order to avoid being divided by 0, use multiplication form:

(x1 - x0) * (y - y1) = (x - x1) * (y1 - y0) =>
dx * (y - y1) = dy * (x - x1), where dx = x1 - x0 and dy = y1 - y0
Now imagine connecting the 2nd points respectively with others one by one, Check if all of the slopes are equal.
'''
class Solution(object):
    def checkStraightLine(self, coordinates):
        """
        :type coordinates: List[List[int]]
        :rtype: bool
        """
        (x0, y0), (x1, y1) = coordinates[: 2]
        for x, y in coordinates:
            if (x1 - x0) * (y - y1) != (x - x1) * (y1 - y0):
                return False
        return True
# Day 6 Array
# 1588. Sum of All Odd Length Subarrays
'''
Solution 2: Consider the contribution of A[i]
Also suggested by @mayank12559 and @simtully.

Consider the subarray that contains A[i],
we can take 0,1,2..,i elements on the left,
from A[0] to A[i],
we have i + 1 choices.

we can take 0,1,2..,n-1-i elements on the right,
from A[i] to A[n-1],
we have n - i choices.

In total, there are k = (i + 1) * (n - i) subarrays, that contains A[i].
And there are (k + 1) / 2 subarrays with odd length, that contains A[i].
And there are k / 2 subarrays with even length, that contains A[i].

A[i] will be counted ((i + 1) * (n - i) + 1) / 2 times for our question.


Example of array [1,2,3,4,5]
1 2 3 4 5 subarray length 1
1 2 X X X subarray length 2
X 2 3 X X subarray length 2
X X 3 4 X subarray length 2
X X X 4 5 subarray length 2
1 2 3 X X subarray length 3
X 2 3 4 X subarray length 3
X X 3 4 5 subarray length 3
1 2 3 4 X subarray length 4
X 2 3 4 5 subarray length 4
1 2 3 4 5 subarray length 5

5 8 9 8 5 total times each index was added, k = (i + 1) * (n - i)
3 4 5 4 3 total times in odd length array with (k + 1) / 2
2 4 4 4 2 total times in even length array with k / 2s


Complexity
Time O(N)
Space O(1)
'''
class Solution(object):
    def sumOddLengthSubarrays(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        res, n = 0, len(arr)
        for i, a in enumerate(arr):
            res += ((i + 1) * (n - i) + 1) / 2 * a
        return res

class Solution(object):
    def sumOddLengthSubarrays(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        return sum(((i + 1) * (len(arr) - i) + 1) / 2 * a for i, a in enumerate(arr))
# 283. Move Zeroes
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        l = 0
        r = len(nums)
        while l < r:
            if nums[l] == 0:
                del nums[l]
                nums.append(0)
                r -= 1
            else:
                l += 1
# 1672. Richest Customer Wealth
'''
Solution: Sum each row up
Time complexity: O(mn)
Space complexity: O(1)'''
class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        return max(sum(account) for account in accounts)
'''
There's a concept in functional programming called Higher-Order Functions, 
which are a type of functions that can accept functions as arguments and/or 
return functions as results. 
The built-in function map is an example of Higher-Order Functions, 
it takes the function sum and applies it to each entry of the accounts list.

Therefore, map(sum, accounts) returns a list containing the sum of each sub-list
 in the accounts list and max(map(sum, accounts)) 
 returns the maximum sum from the sums produced using the map function.
'''
class Solution(object):
    def maximumWealth(self, accounts):
        """
        :type accounts: List[List[int]]
        :rtype: int
        """
        return max(map(sum, accounts))
# Day 7 Array
# 1572. Matrix Diagonal Sum
class Solution(object):
    def diagonalSum(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        n = len(mat)
        mid = n // 2
        summation = 0
        
        for i in range(n):
            # primary diagonal
            summation += mat[i][i]
            # secondary diagonal
            summation += mat[n-1-i][i]
                        
        if n % 2 == 1:
            # remove center element (repeated) on odd side-length case
            summation -= mat[mid][mid]
    
        return summation

class Solution(object):
    def diagonalSum(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        return sum(sum(r[j] for j in {i, len(r) - i - 1}) for i, r in enumerate(mat))

# 566. Reshape the Matrix
import numpy as np
class Solution(object):
    def matrixReshape(self, mat, r, c):
        """
        :type mat: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        m, n = len(mat), len(mat[0])
        if r * c != m * n: return mat  # Invalid size -> return original matrix
        ans = [[0] * c for _ in range(r)]
        for i in range(m * n):
            ans[i // c][i % c] = mat[i // n][i % n]
        return ans


# Day 8 String
# 1768. Merge Strings Alternately
class Solution(object):
    def mergeAlternately(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: str
        """
        res=''
        
        for i in range(min(len(word1),len(word2))):
            res += word1[i] + word2[i]
            
        return res + word1[i+1:] + word2[i+1:]
# Python 3 only ! using zip
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return ''.join(a + b for a, b in zip_longest(word1, word2, fillvalue=''))

# 1678. Goal Parser Interpretation
'''
Solution: String
If we encounter '(' check the next character to determine whether it's '()' or '(al')

Time complexity: O(n)
Space complexity: O(n)
'''
class Solution(object):
    def interpret(self, command):
        """
        :type command: str
        :rtype: str
        """
        ans = []
        for i, c in enumerate(command):
            if c == 'G': ans.append('G')
            elif c == '(':
                ans.append('o' if command[i + 1] == ')' else 'al')
        return ''.join(ans)

# 389. Find the Difference
'''
In this problem we are given 2 strings "t" & "s". example:
s = "abc"
t = "bacx" in "t" we have one additional character

So, we have to find one additional character there are couple of ways to solve this problem.But we will use BITWISE method to solve this problem

So, using XOR concept we will get our additional character, understand it visually :
So, here also let's say our character are:
s = abc
t = cabx

if we take XOR of every character. all the n character of s "abc" is similar to n character of t "cab". So, they will cancel each other. 
And we left with our answer.

s =   abc
t =   cbax
------------
ans -> x
-----------
'''
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        c = 0
        for cs in s: c ^= ord(cs) #ord is ASCII value
        for ct in t: c ^= ord(ct)
        return chr(c) #chr = convert ASCII into character
# Day 9 String
# 709. To Lower Case
'''
Solution
Time complexity: O(n)
Space complexity: O(1)
'''
class Solution(object):
    def toLowerCase(self, s):
        """
        :type s: str
        :rtype: str
        """
        ans = ''
        for c in s:
            if c >= 'A' and c <= 'Z':
                ans += chr(ord(c) + 32)
            else:
                ans += c
        return ans
# 1309. Decrypt String from Alphabet to Integer Mapping
# Regex
class Solution(object):
    def freqAlphabets(self, s):
        """
        :type s: str
        :rtype: str
        """
        return ''.join(chr(int(i[:2]) + 96) for i in re.findall(r'\d\d#|\d', s))

# 953. Verifying an Alien Dictionary
'''Mapping to Normal Order
Explanation
Build a transform mapping from order,
Find all alien words with letters in normal order.

For example, if we have order = "xyz..."
We can map the word "xyz" to "abc" or "123"

Then we check if all words are in sorted order.

Complexity
Time O(NS)
Space O(1)'''
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        m = {c: i for i, c in enumerate(order)}
        words = [[m[c] for c in w] for w in words]
        return all(w1 <= w2 for w1, w2 in zip(words, words[1:]))

# Day 10 Linked List & Tree
# 1290. Convert Binary Number in a Linked List to Integer
# Approach : Bit Manipulation
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def getDecimalValue(self, head):
        """
        :type head: ListNode
        :rtype: int
        """
        num = head.val
        while head.next:
            num = (num << 1) | head.next.val
            head = head.next
        return num
# Approach : Binary Representation
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def getDecimalValue(self, head):
        """
        :type head: ListNode
        :rtype: int
        """
        num = head.val
        while head.next:
            num = num * 2 + head.next.val
            head = head.next
        return num

# 876. Middle of the Linked List
'''
Each time, slow go 1 steps while fast go 2 steps.
When fast arrives at the end, slow will arrive right in the middle.'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # While slow moves one step forward, fast moves two steps forward.
        # Finally, when fast reaches the end, slow happens to be in the middle of the linked list.
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
# 104. Maximum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1

# 404. Sum of Left Leaves
'''
Solution: Recursion
Time complexity: O(n)
Space complexity: O(h)
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if (root==None): return 0
        if (root.left and root.left.left==None and root.left.right==None):
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)
# Iterative
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if (root == None): return 0
        s, ans = deque([root]), 0
        while s:
            cur = s.pop()
            if cur.left!=None:
                if not cur.left.left and not cur.left.right:
                    ans = ans + cur.left.val
                else: 
                    s.append(cur.left)
            if cur.right!=None: 
                s.append(cur.right)
            
        return ans


# Day 11 Containers & Libraries
# 1356. Sort Integers by The Number of 1 Bits
'''
Explanation by @Be_Kind_To_One_Another:
Comment regarding i -> Integer.bitCount(i) * 10000 + i to answer any potential question. 
It's essentially hashing the original numbers into another number generated from the count of bits 
and then sorting the newly generated numbers. 
so why 10000? simply because of the input range is 0 <= arr[i] <= 10^4.
For instance [0,1,2,3,5,7], becomes something like this [0, 10001, 10002, 20003, 20005, 30007].

0 has 0 number of bits  --> 0 * 10000 + 0 = 0
1,2 have 1 bit set      --> 1 * 10000 + 1 = 10001  &  1 * 10000 + 2 = 10002
3,5 have 2 bits set     --> 2 * 10000 + 3 = 20003  &  2 * 10000 + 5 = 20005
7 has 3 bits set        --> 3 * 10000 + 7 = 30007
In short, the bit length contribution to the value of hash code is always greater than the number itself. 
'''
class Solution(object):
    def sortByBits(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        return sorted(arr, key = lambda num : (sum((num >> i) & 1 for i in range(32)), num))

# without using lib to count the bit:
class Solution(object):
    def sortByBits(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        def bit_count(x):
            bit = 0
            while x > 0:
                bit += x % 2
                x //= 2
            return bit
        
        return sorted(arr, key=lambda x: (bit_count(x), x))

# python 3 version
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        return sorted(arr, key=lambda x: (bin(x).count('1') << 16) + x)

# 232. Implement Queue using Stacks
class MyQueue(object):

    def __init__(self):
        self.s1 = []
        self.s2 = []
        self.front = None

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if not self.s1: self.front = x
        self.s1.append(x)
        
    def pop(self):
        """
        :rtype: int
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2.pop()
        
    def peek(self):
        """
        :rtype: int
        """
        if self.s2 :
            return self.s2[-1]
        return self.front
        
    def empty(self):
        """
        :rtype: bool
        """
        return not self.s1 and not self.s2
        
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()

# 242. Valid Anagram
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s) == sorted(t)

# 217. Contains Duplicate
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """ 
        return len(set(nums)) != len(nums)

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """ 
       # if len(nums) <=1:
        #    return False
        nums.sort()
        for i in range(0,len(nums)-1):
            if nums[i] == nums[i+1]:
                return True
        return False

# Day 12 Class & Object
# 1603. Design Parking System
'''
Solution: Simulation
Time complexity: O(1) per addCar call
Space complexity: O(1)'''
class ParkingSystem(object):

    def __init__(self, big, medium, small):
        """
        :type big: int
        :type medium: int
        :type small: int
        """
        self.A = [big, medium, small]
        

    def addCar(self, carType):
        """
        :type carType: int
        :rtype: bool
        """
        self.A[carType - 1] -= 1
        return self.A[carType - 1] >= 0


# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)

class ParkingSystem(object):

    def __init__(self, big, medium, small):
        """
        :type big: int
        :type medium: int
        :type small: int
        """
        self.slots = {1: big, 2: medium, 3: small}
        

    def addCar(self, carType):
        """
        :type carType: int
        :rtype: bool
        """
        if self.slots[carType] == 0: return False
        self.slots[carType] -= 1
        return True


# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)

# 303. Range Sum Query - Immutable
'''
Solution: Prefix sum
sums[i] = nums[0] + nums[1] + … + nums[i]

sumRange(i, j) = sums[j] - sums[i - 1]

Time complexity: pre-compute: O(n), query: O(1)
Space complexity: O(n)'''
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.preSum = nums  # pass by pointer!
        for i in range(len(nums)-1):
            self.preSum[i+1] += self.preSum[i]
        
    def sumRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        if left == 0: return self.preSum[right]
        return self.preSum[right] - self.preSum[left-1]
        
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)