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

# 1- liner
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        return words == sorted(words, key=lambda w: map(order.index, w))

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
'''Solution: Recursion
maxDepth(root) = max(maxDepth(root.left), maxDepth(root.right)) + 1

Time complexity: O(n)
Space complexity: O(n)'''
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

''' Iterative '''
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

# Programming Skills II
# Day 1
# 896. Monotonic Array
class Solution(object):
    def isMonotonic(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        inc = True
        dec = True

        for i in range(1, len(nums)):
            inc = inc and nums[i] >= nums[i - 1]
            dec = dec and nums[i] <= nums[i - 1]    

        return inc or dec

class Solution(object):
    def isMonotonic(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return not {cmp(i, j) for i, j in zip(nums, nums[1:])} >= {1, -1}

# 28. Implement strStr()
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        l1 = len(haystack)
        l2 = len(needle)
        for i in range(l1 - l2 + 1):
            if haystack[i:i+l2] == needle: return i
        return -1

# Day 2
# 110. Balanced Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.balanced = True
        def height(root):
            if not root or not self.balanced: return -1
            l = height(root.left)
            r = height(root.right)
            if abs(l - r) > 1:
                self.balanced = False
                return -1
            return max(l, r) + 1
        height(root)
        return self.balanced

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return True
        self.balanced = True
       
        def height(root,balanced):
            if not root: return 0
            left_height = height(root.left, self.balanced)
            if not self.balanced:return -1 
            right_height = height(root.right, self.balanced)
            if not self.balanced:return -1
            if abs(left_height-right_height) > 1:
                self.balanced = False
                return -1
            return max(left_height,right_height) +1
        height(root,self.balanced)
        return self.balanced

# 459. Repeated Substring Pattern
'''
Basic idea:

First char of input string is first char of repeated substring
Last char of input string is last char of repeated substring
Let S1 = S + S (where S in input string)
Remove 1 and last char of S1. Let this be S2
If S exists in S2 then return true else false
Let i be index in S2 where S starts then repeated substring length i + 1 
and repeated substring S[0: i+1]
'''
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not str:
            return False
            
        ss = (s + s)[1:-1]
        return ss.find(s) != -1

# Day 3
# 150. Evaluate Reverse Polish Notation
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for t in tokens:
            if t not in "+-*/":
                stack.append(int(t))
            else:
                r, l = stack.pop(), stack.pop()
                if t == "+":
                    stack.append(l+r)
                elif t == "-":
                    stack.append(l-r)
                elif t == "*":
                    stack.append(l*r)
                else:# You can use int(float(l) / r) for division instead of` if statement. 
                #int()will take the part of integer, which is wanted by this question.
                    stack.append(int(float(l)/r))
        return stack.pop()

# 66. Plus One
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if len(digits) == 0:
            digits = [1]
        elif digits[-1] == 9:
            digits = self.plusOne(digits[:-1])
            digits.extend([0])
        else:
            digits[-1] += 1
        return digits

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        return map(int, list(str(int(''.join(map(str, digits))) + 1)))

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        for i in range(len(digits)):
            if digits[~i] < 9:
                digits[~i] += 1
                return digits
            digits[~i] = 0
        return [1] + [0] * len(digits)

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        length = len(digits) - 1
        while digits[length] == 9:
            digits[length] = 0
            length -= 1
        if(length < 0):
            digits = [1] + digits
        else:
            digits[length] += 1
        return digits

# Day 4
# 1367. Linked List in Binary Tree
'''
Solution 1: Brute DFS
Time O(N * min(L,H))
Space O(H)
where N = tree size, H = tree height, L = list length.
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubPath(self, head, root):
        """
        :type head: ListNode
        :type root: TreeNode
        :rtype: bool
        """
        def dfs(head, root):
            if not head: return True
            if not root: return False
            return root.val == head.val and (dfs(head.next, root.left) or dfs(head.next, root.right))
        if not head: return True
        if not root: return False
        return dfs(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)
'''
Solution 2: DP
Iterate the whole link, find the maximum matched length of prefix.
Iterate the whole tree, find the maximum matched length of prefix.
About this dp, @fukuzawa_yumi gave a link of reference:
https://en.wikipedia.org/wiki/Knuth–Morris–Pratt_algorithm

Time O(N + L)
Space O(L + H)
where N = tree size, H = tree height, L = list length.'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubPath(self, head, root):
        """
        :type head: ListNode
        :type root: TreeNode
        :rtype: bool
        """
        A, dp = [head.val], [0]
        i = 0
        node = head.next
        while node:
            while i and node.val != A[i]:
                i = dp[i - 1]
            i += node.val == A[i]
            A.append(node.val)
            dp.append(i)
            node = node.next

        def dfs(root, i):
            if not root: return False
            while i and root.val != A[i]:
                i = dp[i - 1]
            i += root.val == A[i]
            return i == len(dp) or dfs(root.left, i) or dfs(root.right, i)
        return dfs(root, 0)

# 43. Multiply Strings

# Day 5
# 67. Add Binary
# https://leetcode.com/problems/add-binary/discuss/1679423/Well-Detailed-Explaination-Java-C%2B%2B-Python-oror-Easy-for-mind-to-Accept-it
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        res = ""
        i, j, carry = len(a) - 1, len(b) - 1, 0
        while i >= 0 or j >= 0:
            sum = carry
            if i >= 0 : sum += ord(a[i]) - ord('0') # ord is use to get value of ASCII character
            if j >= 0 : sum += ord(b[j]) - ord('0')
            i, j = i - 1, j - 1
            carry = 1 if sum > 1 else 0
            res += str(sum % 2)

        if carry != 0 : res += str(carry)
        return res[::-1]

# 989. Add to Array-Form of Integer
'''Take K itself as a Carry
Explanation
Take K as a carry.
Add it to the lowest digit,
Update carry K,
and keep going to higher digit.


Complexity
Insert will take O(1) time or O(N) time on shifting, depending on the data stucture.
But in this problem K is at most 5 digit so this is restricted.
So this part doesn't matter.

The overall time complexity is O(N).
For space I'll say O(1)

With one loop.'''
class Solution(object):
    def addToArrayForm(self, num, k):
        """
        :type num: List[int]
        :type k: int
        :rtype: List[int]
        """
        for i in range(len(num) - 1, -1, -1):
            k, num[i] = divmod(num[i] + k, 10)
        return [int(i) for i in str(k)] + num if k else num

# Day 6
# 739. Daily Temperatures
class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        st = []
        n = len(temperatures)
        ans = [0] * n
        for i in range(n - 1, -1, -1):
            t = temperatures[i]
            while st and temperatures[st[-1]] <= t:
                st.pop()
            if st:
                ans[i] = (st[-1] - i)
            st.append(i)

        return ans

# 58. Length of Last Word
'''
We can just split our string, remove all extra spaces 
and return length of the last word, 
however we need to spend O(n) time for this, where n is length of our string. 
There is a simple optimization: let us traverse string from the end and:

find the last element of last word: 
traverse from the end and find first non-space symbol.
continue traverse and find first space symbol (or beginning of string)
return end - beg.
Complexity: is O(m), 
where m is length of part from first symbol of last word to the end. 
Space complexity is O(1).
'''
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        end = len(s) - 1
        while end > 0 and s[end] == " ": end -= 1
        beg = end
        while beg >= 0 and s[beg] != " ": beg -= 1
        return end - beg

# one - liner
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        return 0 if not s.split() else len(s.split()[-1])
# Day 7
# 48. Rotate Image
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix[0])
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp

# 1886. Determine Whether Matrix Can Be Obtained By Rotation
'''
In order to rotate clockwise, first reverse rows order, then transpose the matrix;
Rotate 0, 1, 2, 3 times and compare the rotated matrix with target, respectively.'''
class Solution(object):
    def findRotation(self, mat, target):
        """
        :type mat: List[List[int]]
        :type target: List[List[int]]
        :rtype: bool
        """
        for _ in range(4): 
            if mat == target: return True
            mat = [list(x) for x in zip(*mat[::-1])]
        return False 

# Day 8
# 54. Spiral Matrix
# The con is mutating the matrix, if this is not allowed, 
# we can make a deep copy of the matrix first. 
# And of course it comes with the additional memory usage.
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        ret = []
        while matrix:
            ret += matrix.pop(0)
            if matrix and matrix[0]:
                for row in matrix:
                    ret.append(row.pop())
            if matrix:
                ret += matrix.pop()[::-1]
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    ret.append(row.pop(0))
        return ret

# 973. K Closest Points to Origin

# Day 9
# 1630. Arithmetic Subarrays
class Solution(object):
    def checkArithmeticSubarrays(self, nums, l, r):
        """
        :type nums: List[int]
        :type l: List[int]
        :type r: List[int]
        :rtype: List[bool]
        """
        ans = []
        
        for i , j in zip(l , r):
            arr = nums[i:j + 1]
            arr.sort()
            dif = []
            
            for i in range(len(arr) - 1):
                dif.append(arr[i] - arr[i + 1])
            
            ans.append(len(set(dif)) == 1)
        
        return ans

class Solution(object):
    def checkArithmeticSubarrays(self, nums, l, r):
        """
        :type nums: List[int]
        :type l: List[int]
        :type r: List[int]
        :rtype: List[bool]
        """
        ans = []
        
        def find_diffs(arr):
            
            arr.sort()

            dif = []
            
            for i in range(len(arr) - 1):
                dif.append(arr[i] - arr[i + 1])
            
            return len(set(dif)) == 1
        
        for i , j in zip(l , r):
            ans.append(find_diffs(nums[i:j + 1]))
        
        return ans

# 429. N-ary Tree Level Order Traversal
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        if root == None: return []
        q = deque([root])
        ans = []
        while q:
            level = []
            for _ in range(len(q)):
                curr = q.popleft()
                level.append(curr.val)
                for child in curr.children:
                    q.append(child)
            ans.append(level)
        return ans

# Day 10
# 503. Next Greater Element II
'''
Loop Twice
Explanation
Loop once, we can get the Next Greater Number of a normal array.
Loop twice, we can get the Next Greater Number of a circular array


Complexity
Time O(N) for one pass
Spce O(N) in worst case
'''
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        stack, res = [], [-1] * len(nums)
        for i in range(len(nums)) * 2:
            while stack and (nums[stack[-1]] < nums[i]):
                res[stack.pop()] = nums[i]
            stack.append(i)
        return res

# 556. Next Greater Element III
# Next Permutation 
class Solution(object):
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        digits = list(str(n))
        length = len(digits)
        
        i, j = length-2, length-1
        while i >= 0 and digits[i+1] <= digits[i]:
            i -= 1
        
        if i == -1: return -1

        while digits[j] <= digits[i]:
            j -= 1
        
        digits[i], digits[j] = digits[j], digits[i]

        res = int(''.join(digits[:i+1] + digits[i+1:][::-1]))
        if res >= 2**31 or res == n:
            return -1
        return res

# Day 11
# 1376. Time Needed to Inform All Employees

# 49. Group Anagrams

# Day 12
# 438. Find All Anagrams in a String

# 713. Subarray Product Less Than K

# Day 13
# 304. Range Sum Query 2D - Immutable

# 910. Smallest Range II
'''
Intuition:
For each integer A[i],
we may choose either x = -K or x = K.

If we add K to all B[i], the result won't change.

It's the same as:
For each integer A[i], we may choose either x = 0 or x = 2 * K.

Explanation:
We sort the A first, and we choose to add x = 0 to all A[i].
Now we have res = A[n - 1] - A[0].
Starting from the smallest of A, we add 2 * K to A[i],
hoping this process will reduce the difference.

Update the new mx = max(mx, A[i] + 2 * K)
Update the new mn = min(A[i + 1], A[0] + 2 * K)
Update the res = min(res, mx - mn)

Time Complexity:
O(NlogN), in both of the worst and the best cases.
In the Extending Reading part, I improve this to O(N) in half of cases.
'''
class Solution(object):
    def smallestRangeII(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort()
        res = nums[-1] - nums[0]
        for i in range(len(nums) - 1):
            big = max(nums[-1], nums[i] + 2 * k)
            small = min(nums[i + 1], nums[0] + 2 * k)
            res = min(res, big - small)
        return res

# Day 14
# 143. Reorder List

# 138. Copy List with Random Pointer
"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        cache = {}
        new_ptr = dummy = Node(-1)
        
        while head:
            if head in cache: newnode = cache[head]
            else:
                newnode = Node(head.val)
                cache[head] = newnode
            new_ptr.next = newnode
            new_ptr = new_ptr.next
            if head.random:    
                if head.random in cache: new_random = cache[head.random]
                else:
                    new_random = Node(head.random.val)
                    cache[head.random] = new_random
                new_ptr.random = new_random
            head = head.next
        return dummy.next

# Day 15
# 2. Add Two Numbers

# 445. Add Two Numbers II
'''
Since there is no maximum of int in python, 
we can computer the sum and then construct the result link list. 
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        x1, x2 = 0, 0
        while l1:
            x1 = x1*10+l1.val
            l1 = l1.next
        while l2:
            x2 = x2*10+l2.val
            l2 = l2.next
        x = x1 + x2
        
        head = ListNode(0)
        if x == 0: return head
        while x:
            v, x = x%10, x//10
            head.next, head.next.next = ListNode(v), head.next
            
        return head.next

# Day 16
# 61. Rotate List
'''
Solution: Find the prev of the new head

Step 1: Get the tail node T while counting the length of the list.
Step 2: k %= l, k can be greater than l, rotate k % l times has the same effect.
Step 3: Find the previous node P of the new head N by moving (l – k – 1) steps from head
Step 4: set P.next to null, T.next to head and return N

Time complexity: O(n) n is the length of the list
Space complexity: O(1)
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head: return head
        tail = head
        l = 1
        while tail.next: 
            tail = tail.next
            l += 1
        k = k % l
        if k == 0: return head

        prev = head
        for _ in range(l - k - 1): prev = prev.next

        new_head = prev.next
        tail.next = head
        prev.next = None
        return new_head

# 173. Binary Search Tree Iterator

# Day 17
# 1845. Seat Reservation Manager
'''
Use heap to save available seats

Time: O(N) for init, O(logN) for other functions
Space: O(N)'''
class SeatManager(object):

    def __init__(self, n):
        """
        :type n: int
        """
        self.heap = list(range(1, n + 1))

    def reserve(self):
        """
        :rtype: int
        """
        return heapq.heappop(self.heap)
        

    def unreserve(self, seatNumber):
        """
        :type seatNumber: int
        :rtype: None
        """
        heapq.heappush(self.heap, seatNumber)


# Your SeatManager object will be instantiated and called as such:
# obj = SeatManager(n)
# param_1 = obj.reserve()
# obj.unreserve(seatNumber)

# 860. Lemonade Change
'''
Intuition:
When the customer gives us $20, we have two options:

To give three $5 in return
To give one $5 and one $10.
On insight is that the second option (if possible) is always better 
than the first one.
Because two $5 in hand is always better than one $10


Explanation:
Count the number of $5 and $10 in hand.

if (customer pays with $5) five++;
if (customer pays with $10) ten++, five--;
if (customer pays with $20) ten--, five-- or five -= 3;

Check if five is positive, otherwise return false.


Time Complexity
Time O(N) for one iteration
Space O(1)
'''
class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        five = ten = 0
        for i in bills:
            if i == 5: five += 1
            elif i == 10: five, ten = five - 1, ten + 1
            elif ten > 0: five, ten = five - 1, ten - 1
            else: five -= 3
            if five < 0: return False
        return True

# Day 18
# 155. Min Stack

# 341. Flatten Nested List Iterator
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = nestedList[::-1]

    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop().getInteger()
        
'''we can simply pop off the top element and the extend it with the new inner lists: 
self.stack.extend(self.stack.pop().getList()[::-1]). 
We also reduce a line because there is no need to define top'''
    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
            if self.stack[-1].isInteger():
                return True
            self.stack.extend(self.stack.pop().getList()[::-1])
        return False
        

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

# Day 19
# 1797. Design Authentication Manager
'''
Solution: Hashtable
Use a hashtable to store the token and its expiration time.

Time complexity: at most O(n) per operation
Space complexity: O(n)
'''
class AuthenticationManager(object):

    def __init__(self, timeToLive):
        """
        :type timeToLive: int
        """
        self.token = dict()
        self.time = timeToLive    # store timeToLive and create dictionary
        

    def generate(self, tokenId, currentTime):
        """
        :type tokenId: str
        :type currentTime: int
        :rtype: None
        """
        self.token[tokenId] = currentTime    # store tokenId with currentTime
        

    def renew(self, tokenId, currentTime):
        """
        :type tokenId: str
        :type currentTime: int
        :rtype: None
        """
        limit = currentTime-self.time        # calculate limit time to filter unexpired tokens
        if tokenId in self.token and self.token[tokenId]>limit:    # filter tokens and renew its time
            self.token[tokenId] = currentTime
        

    def countUnexpiredTokens(self, currentTime):
        """
        :type currentTime: int
        :rtype: int
        """
        limit = currentTime-self.time       # calculate limit time to filter unexpired tokens
        c = 0
        for i in self.token:
            if self.token[i]>limit:         # count unexpired tokens
                c+=1
        return c


# Your AuthenticationManager object will be instantiated and called as such:
# obj = AuthenticationManager(timeToLive)
# obj.generate(tokenId,currentTime)
# obj.renew(tokenId,currentTime)
# param_3 = obj.countUnexpiredTokens(currentTime)

# 707. Design Linked List

# Day 20
# 380. Insert Delete GetRandom O(1)
'''
In python, creating a simple api for a set() would be a perfect solution 
if not for the third operation, getRandom(). 
We know that we can retrieve an item from a set, 
and not know what that item will be, but that would not be actually random. 
(This is due to the way python implements sets. 
In python3, when using integers, elements are popped from the set in the order 
they appear in the underlying
hashtable. Hence, not actually random.)

A set is implemented essentially the same as a dict in python, 
so the time complexity of add / delete is on average O(1). 
When it comes to the random function, however, we run into the problem of 
needing to convert the data into a python list in order to return 
a random element. That conversion will add a significant overhead to getRandom,
 thus slowing the whole thing down.

Instead of having to do that type conversion (set to list) 
we can take an approach that involves maintaining both a list and a dictionary 
side by side. That might look something like:

data_map == {4: 0, 6: 1, 2: 2, 5: 3}
data == [4, 6, 2, 5]
Notice that the key in the data_map is the element in the list, 
and the value in the data_map is the index the element is at in the list.
Notes:

this can be made more efficient by removing the variables last_elem_in_list and index_of_elem_to_remove. 
I have used this to aid in readability.
the remove operation might appear complicated so here's a before and after of what the data looks like:
element_to_remove = 7

before:     [4, 7, 9, 3, 5]
after:      [4, 5, 9, 3]

before:     {4:0, 7:1, 9:2, 3:3, 5:4}
after:      {4:0, 9:2, 3:3, 5:1}
All we're doing is replacing the element in the list that needs to be removed with the last element in the list.
And then we update the values in the dictionary to reflect that.'''
class RandomizedSet(object):

    def __init__(self):
        self.data_map = {} # dictionary, aka map, aka hashtable, aka hashmap
        self.data = [] # list aka array

    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        # the problem indicates we need to return False if the item 
        # is already in the RandomizedSet---checking if it's in the
        # dictionary is on average O(1) where as
        # checking the array is on average O(n)
        if val in self.data_map:
            return False
        
        # add the element to the dictionary. Setting the value as the 
        # length of the list will accurately point to the index of the 
        # new element. (len(some_list) is equal to the index of the last item +1)
        self.data_map[val] = len(self.data)

        # add to the list
        self.data.append(val)
        
        return True
        

    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        # again, if the item is not in the data_map, return False. 
        # we check the dictionary instead of the list due to lookup complexity
        if not val in self.data_map:
            return False

        # essentially, we're going to move the last element in the list 
        # into the location of the element we want to remove. 
        # this is a significantly more efficient operation than the obvious 
        # solution of removing the item and shifting the values of every item 
        # in the dicitionary to match their new position in the list
        last_elem_in_list = self.data[-1]
        index_of_elem_to_remove = self.data_map[val]

        self.data_map[last_elem_in_list] = index_of_elem_to_remove
        self.data[index_of_elem_to_remove] = last_elem_in_list

        # change the last element in the list to now be the value of the element 
        # we want to remove
        self.data[-1] = val

        # remove the last element in the list
        self.data.pop()

        # remove the element to be removed from the dictionary
        self.data_map.pop(val)
        return True
        

    def getRandom(self):
        """
        :rtype: int
        """
        # if running outside of leetcode, you need to `import random`.
        # random.choice will randomly select an element from the list of data.
        return random.choice(self.data)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# 622. Design Circular Queue
'''
Solution: Simulate with an array
We need a fixed length array, 
and the head location as well as the size of the current queue.

We can use q[head] to access the front, 
and q[(head + size – 1) % k] to access the rear.

Time complexity: O(1) for all the operations.
Space complexity: O(k)
'''
class MyCircularQueue(object):

    def __init__(self, k):
        """
        :type k: int
        """
        self.q = [0] * k
        self.k = k
        self.head = self.size = 0
        

    def enQueue(self, value):
        """
        :type value: int
        :rtype: bool
        """
        if self.isFull(): return False
        self.q[(self.head + self.size) % self.k] = value
        self.size += 1
        return True

    def deQueue(self):
        """
        :rtype: bool
        """
        if self.isEmpty(): return False
        self.head = (self.head + 1) % self.k
        self.size -= 1
        return True
        

    def Front(self):
        """
        :rtype: int
        """
        return -1 if self.isEmpty() else self.q[self.head]


    def Rear(self):
        """
        :rtype: int
        """
        return -1 if self.isEmpty() else self.q[(self.head + self.size - 1) % self.k]
        

    def isEmpty(self):
        """
        :rtype: bool
        """
        return self.size == 0

    def isFull(self):
        """
        :rtype: bool
        """
        return self.size == self.k


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()

'''
Complexity:

Time: Each operation take O(1)
Space: O(k)'''
class MyCircularQueue(object):

    def __init__(self, k):
        """
        :type k: int
        """
        self.head = self.tail = self.size = 0
        self.arr = [0] * k
        

    def enQueue(self, value):
        """
        :type value: int
        :rtype: bool
        """
        if self.isFull(): return False
        self.arr[self.tail] = value
        self.tail = (self.tail + 1) % len(self.arr)
        self.size += 1
        return True

    def deQueue(self):
        """
        :rtype: bool
        """
        if self.isEmpty(): return False
        self.head = (self.head + 1) % len(self.arr)
        self.size -= 1
        return True
        

    def Front(self):
        """
        :rtype: int
        """
        if self.isEmpty(): return -1
        return self.arr[self.head]


    def Rear(self):
        """
        :rtype: int
        """
        if self.isEmpty(): return -1
        return self.arr[self.tail-1]
        

    def isEmpty(self):
        """
        :rtype: bool
        """
        return self.size == 0

    def isFull(self):
        """
        :rtype: bool
        """
        return self.size == len(self.arr)


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()

# 729. My Calendar I
class MyCalendar(object):

    def __init__(self):
         self.calendar = []

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        for s, e in self.calendar:
            if s < end and start < e:
                return False
        self.calendar.append((start, end))
        return True


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)