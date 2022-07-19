# Day 1
# 704. Binary Search
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l,r = 0, len(nums)
        while(l<r):
            m = l+(r-l)//2
            if nums[m] == target:
                return m
            elif nums[m] > target :
                r = m
            else:
                l = m+1
        return -1

# 374. Guess Number Higher or Lower

# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        l,r=0,n
        while(l<r):
            m = l+(r-l)//2
            if guess(m) == 0:
                return m
            elif guess(m) == 1:
                l = m+1
            else:
                r = m
        return l

# Day 2
# 35. Search Insert Position
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l,r = 0, len(nums)
        while(l<r):
            m = l+(r-l)//2
            if nums[m] == target:
                return m
            elif nums[m] > target:
                r = m
            else:
                l = m+1
        return l

# 852. Peak Index in a Mountain Array
class Solution(object):
    def peakIndexInMountainArray(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        l,r = 0, len(arr)
        while(l<r):
            m = l+(r-l)//2
            if arr[m] > arr[m+1]:
                r = m
            else:
                l = m+1
        return l

# Day 3 
# 367. Valid Perfect Square
# Binary Search
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num ==1:
            return True
        l,r = 0,num
        while l < r:
            m = l+(r-l)//2
            if m**2 == num:
                return True
            elif m**2 > num:
                r = m
            else:
                l = m+1
        return False

# a slightly better option
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        l = 0
        r = num
        
        while l <= r:
            m = l + (r-l)//2
            if  m ** 2 == num:
                return True
            elif m ** 2 > num:
                r = m -1
            else:
                l = m +1
        return False

# 1385. Find the Distance Value Between Two Arrays
# Binary Search
class Solution(object):
    def findTheDistanceValue(self, arr1, arr2, d):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :type d: int
        :rtype: int
        """
        arr2.sort()
        
        def is_valid(val):
            l, r = 0, len(arr2)
            while l < r:
                mid = (l + r) // 2
                if abs(arr2[mid] - val) <= d:
                    return False
                elif arr2[mid] > val:
                    r = mid
                else:
                    l = mid + 1
            return True

        return sum(is_valid(x) for x in arr1)

# build-in function
class Solution(object):
    def findTheDistanceValue(self, arr1, arr2, d):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :type d: int
        :rtype: int
        """
        return sum(all(abs(a1 - a2) > d for a2 in arr2) for a1 in arr1)

# build-in function
class Solution(object):
    def findTheDistanceValue(self, arr1, arr2, d):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :type d: int
        :rtype: int
        """
        count = 0
        for val1 in arr1:
            if all(abs(val1 - val2) > d for val2 in arr2):
                count += 1

        return count


# Day 4
# 69. Sqrt(x)
# Solution : Newton’s method
class Solution:
    def mySqrt(self, x: int) -> int:
        a = x
        while a * a > x:
          a = (a + x // a) // 2
        return a

# solution 1:
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0 : return 0
        l, r = 1, x  # l have to be 1 b/c m could not be 0
        while l < r:
            m = l + (r - l) / 2
            if m <= x/m and m+1 > x/(m+1):
                return m
            elif m > x/m:
                r = m
            else:
                l = m+1
        return l

# solution 2:
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        l, r = 1, x
        while l <= r:
            m = l + (r - l) // 2
            if m > x/m:
                r = m-1
            else:
                l = m+1
        return r

# 744. Find Smallest Letter Greater Than Target
# binary search typical solution
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        if target >= letters[-1] or target < letters[0]:
            return letters[0]
        
        l, r = 0, len(letters)
        while l < r:
            m = l+(r-l)//2
            # in binary search this would be only greater than
            if  target < letters[m]:
                r = m
            else: l = m+1
        return letters[l]

class Solution(object):
    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        if target >= letters[-1] or target < letters[0]:
            return letters[0]
        
        l, r = 0, len(letters)-1
        while l <= r:
            m = l+(r-l)//2
            # in binary search this would be only greater than
            if  target >= letters[m]: 
                l = m+1
            elif target < letters[m]:
                r = m-1                
        return letters[l]

class Solution(object):
    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        l, r = 0, len(letters)
        while l < r:
            m = l+(r-l)//2
            # in binary search this would be only greater than
            if  target < letters[m]:
                r = m
            else: l = m+1
        # if our insertion position says to insert target into the last position letters.length,
        # we return letters[0] instead. 
        # This is what the modulo operation does.
        return letters[l%len(letters)]

# Day 5
# 278. First Bad Version
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        l,r = 0, n
        while(l<r):
            m = l+(r-l)/2
            if isBadVersion(m)==True:
                r = m
            elif isBadVersion(m)==False :
                l = m + 1      
        return l

# 34. Find First and Last Position of Element in Sorted Array
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        def search_left(nums, target):
            l, r = 0, len(nums)
            while l <r:
                m = l+(r-l)//2
                if nums[m] >= target:
                    r = m
                else:
                    l = m+1

            if l == len(nums) or nums[l] != target: return -1
            return l

        def search_right(nums, target):
            l, r = 0, len(nums)
            while l <r:
                m = l+(r-l)//2
                if nums[m] > target:
                    r = m
                else:
                    l = m+1
            l-=1
            if l < 0 or nums[l] != target: return -1
            return l

        return [search_left(nums, target), search_right(nums, target)]

class Solution(object):    
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def search_left(nums, target):
            ans = -1
            start = 0
            end = len(nums)-1
            while start <=end:
                mid = (start+end)//2
                if nums[mid] == target:
                    ans = mid
                if nums[mid] >= target:
                    end = mid-1
                else:
                    start = mid+1
                    
            return ans
        
        def search_right(nums, target):
            ans= -1
            start = 0
            end = len(nums)-1
            while start<=end:
                mid = (start+end)//2
                if nums[mid]==target:
                    ans = mid
                    
                if nums[mid] <= target:
                    start = mid+1
                    
                else:
                    end = mid-1
            return ans
        
        return [search_left(nums, target), search_right(nums, target)]

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def search_left(nums, target):
            ans = -1
            start = 0
            end = len(nums)
            while start <end:
                mid = start+(end-start)//2
                if nums[mid] == target:
                    ans = mid
                if nums[mid] >= target:
                    end = mid
                else:
                    start = mid+1
                    
            return ans
        
        def search_right(nums, target):
            ans= -1
            start = 0
            end = len(nums)
            while start<end:
                mid = start+(end-start)//2
                if nums[mid]==target:
                    ans = mid
                    
                if nums[mid] <= target:
                    start = mid+1
                    
                else:
                    end = mid
            return ans
        
        return [search_left(nums, target), search_right(nums, target)]

# Day 6
# 441. Arranging Coins
class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        l, r = 0, n
        while l <= r:
            m = l + (r - l) // 2
            curr = m * (m + 1) // 2
            if curr == n:
                return m
            if n < curr:
                r = m - 1
            else:
                l = m + 1
        return r

# 1539. Kth Missing Positive Number
'''
Explanation
Assume the final result is x,
And there are m number not missing in the range of [1, x].
Binary search the m in range [0, A.size()].

If there are m number not missing,
that is A[0], A[1] .. A[m-1],
the number of missing under A[m] is A[m] - 1 - m.

If A[m] - 1 - m < k, m is too small, we update left = m.
If A[m] - 1 - m >= k, m is big enough, we update right = m.

Note that, we exit the while loop, l = r,
which equals to the number of missing number used.
So the Kth positive number will be l + k.
'''
class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        l, r = 0, len(arr)
        while l < r:
            m = l + (r - l) // 2
            if arr[m] - 1 - m < k:
                l = m + 1
            else:
                r = m
        return l + k

# Day 7
# 167. Two Sum II - Input Array Is Sorted
# two-pointer
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        l,r = 0, len(numbers)-1
        while(l<r):
            if numbers[l]+numbers[r]==target:
                return [l+1,r+1]
            elif numbers[l]+numbers[r]>target:
                r-=1
            else:
                l+=1
        return []

# Binary Search
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(numbers)):
            l, r = i+1, len(numbers)-1
            tmp = target - numbers[i]
            while l <= r:
                mid = l + (r-l)//2
                if numbers[mid] == tmp:
                    return [i+1, mid+1]
                elif numbers[mid] < tmp:
                    l = mid+1
                else:
                    r = mid-1

# 1608. Special Array With X Elements Greater Than or Equal X
'''
Concept is similar to H-index
After while loop, we can get i which indicates there are already i items larger or equal to i.
Meanwhile, if we found i == nums[i], there will be i + 1 items larger or equal to i, which makes array not special.
Time: O(sort), can achieve O(N) if we use counting sort
Space: O(sort)
'''
class Solution(object):
    def specialArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort(reverse=True)
        l, r = 0, len(nums)
        while l < r:
            m = l + (r - l) // 2
            if m < nums[m]:
                l = m + 1
            else:
                r = m       
        return -1 if l < len(nums) and l == nums[l] else l

# We want to find an i such that nums[i-1] and nums[i] bracket i? Alternative implementation
class Solution(object):
    def specialArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort() 
        if len(nums) <= nums[0]: return len(nums) # edge case 
        for i in range(1, len(nums)): 
            if nums[i-1] < len(nums)-i <= nums[i]: return len(nums)-i
        return -1

# Day 8
# 1351. Count Negative Numbers in a Sorted Matrix
# Binary Search
class Solution(object):
    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return sum([self.binary_search(arr) for arr in grid])
        
    def binary_search(self, arr):
        l, r = 0, len(arr)
        while l < r:
            m = l + (r - l) // 2
            if arr[m] < 0:
                r = m
            else: 
                l = m + 1
        return len(arr) - r

# other solution
class Solution(object):
    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return sum(a < 0 for r in grid for a in r)

class Solution(object):
    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return (np.array(grid) < 0).sum()

class Solution(object):
    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return str(grid).count('-')

# 74. Search a 2D Matrix
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        Treat the 2D array as a 1D array. 
        matrix[index / cols][index % cols]

        Time complexity: O(log(m*n))
        Space complexity: O(1)
        '''
        if not matrix :
            return False
        
        l,r = 0,len(matrix) * len(matrix[0])
        
        while l < r:
            m = l+(r-l)//2
            
            if target == matrix[m/len(matrix[0])][m%len(matrix[0])]:
                return True
            elif matrix[m/len(matrix[0])][m%len(matrix[0])] > target:
                r = m
                
            else:
                l = m+1
                
        return False

# Day 9
# 1337. The K Weakest Rows in a Matrix
# Binary Search
class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """
        temp = sorted([(self.binary_search(row), i) for i, row in enumerate(mat)]) # Is this M*log(M) where M is no of row?
        return [i for v, i in temp][:k]
    def binary_search(self, row): # this will be O(logN) where N is no of col
            l, r = 0, len(row)
            while l < r:
                m = l + (r - l)//2
                if row[m]: l = m + 1
                else: r = m
            return l

class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """
        sums = []
        for i, row in enumerate(mat):
            sums.append((sum(row), i))

        sorted_sums = sorted(sums)

        k_rows = sorted_sums[:k]

        res = []
        for val in k_rows:
            res.append(val[1])

        return res

# 1346. Check If N and Its Double Exist
# Binary Search
class Solution(object):
    def checkIfExist(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        arr = sorted(arr)
        cnt = 0
        for x in arr:
            if x != 0:
                if self.binarySearch(x, arr) and self.binarySearch(x*2, arr):
                    return True
            else: cnt += 1
            
        return cnt >= 2
    
    def binarySearch(self, x, row): # this will be O(logN) where N is no of col
            l, r = 0, len(row)
            while l < r:
                m = l + (r - l)//2
                if row[m] == x: return True 
                elif row[m] > x: r = m
                else: l = m +1
            return False
            
class Solution(object):
    def checkIfExist(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        seen = set()
        for i in arr:
            if 2 * i in seen or i % 2 == 0 and i // 2 in seen:
            # if 2 * i in seen or i / 2 in seen: # credit to @PeterBohai
                return True
            seen.add(i)
        return False

# Day 10
# 350. Intersection of Two Arrays II
''' Approach : Sort then Two Pointers
Complexity:
Time: O(MlogM + NlogN), where M <= 1000 is length of nums1 array, N <= 1000 is length of nums2 array.
Extra Space (without counting output as space): O(1)
'''
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        nums1.sort() # nums1 = sorted(nums1)
        nums2.sort()
        
        ans = []
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                ans.append(nums1[i])
                i += 1
                j += 1
        return ans
# use dictionary to count
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        counts = {}
        res = []

        for num in nums1:
            counts[num] = counts.get(num, 0) + 1

        for num in nums2:
            if num in counts and counts[num] > 0:
                res.append(num)
                counts[num] -= 1

        return res
# use Counter to make it cleaner
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        counts = collections.Counter(nums1)
        res = []

        for num in nums2:
            if counts[num] > 0:
                res += num,
                counts[num] -= 1

        return res

class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        c1, c2 = collections.Counter(nums1), collections.Counter(nums2)
        
        for n in c1:
            if n in c2: res.extend([n]*min(c1[n], c2[n]))
        return res
'''
Approach : HashMap
Using HashMap to store occurrences of elements in the nums1 array.
Iterate x in nums2 array, check if cnt[x] > 0 then append x to our answer and decrease cnt[x] by one.
To optimize the space, we ensure len(nums1) <= len(nums2) by swapping nums1 with nums2 if len(nums1) > len(nums2).
Complexity:
Time: O(M + N), where M <= 1000 is length of nums1 array, N <= 1000 is length of nums2 array.
Space: O(min(M, N))
'''
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1) > len(nums2): return self.intersect(nums2, nums1)
            
        cnt = Counter(nums1)
        ans = []
        for x in nums2:
            if cnt[x] > 0:
                ans.append(x)
                cnt[x] -= 1
        return ans
'''  
Follow-up Question 1: What if the given array is already sorted? How would you optimize your algorithm?

Approach 2 is the best choice since we skip the cost of sorting.
So time complexity is O(M+N) and the space complexity is O(1).

Follow-up Question 2: What if nums1's size is small compared to nums2's size? Which algorithm is better?

Approach 1 is the best choice.
Time complexity is O(M+N) and the space complexity is O(M), where M is length of nums1, N is length of nums2.

Follow-up Question 3: What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

If nums1 fits into the memory, we can use Approach 1 which stores all elements of nums1 in the HashMap. Then, we can sequentially load and process nums2.
If neither nums1 nor nums2 fits into the memory, we split the numeric range into numeric sub-ranges that fit into the memory.
We modify Approach 1 to count only elements which belong to the given numeric sub-range.
We process each numeric sub-ranges one by one, util we process all numeric sub-ranges.
For example:
Input constraint:
1 <= nums1.length, nums2.length <= 10^10.
0 <= nums1[i], nums2[i] < 10^5
Our memory can store up to 1000 elements.
Then we split numeric range into numeric sub-ranges [0...999], [1000...1999], ..., [99000...99999], then call Approach 1 to process 100 numeric sub-ranges.
'''
# 633. Sum of Square Numbers
# Two Pointer
'''
Solution : Two Pointers
We can use Two Pointers to search a pair of (left, right), so that left^2 + right^2 = c.
Start with left = 0, right = int(sqrt(c)).
while left <= right:
Let cur = left^2 + right^2.
If cur == c then we found a perfect match -> return True
Else if cur < c, we need to increase cur, so left += 1.
Else we need to decrease cur, so right -= 1.

Complexity
Time: O(sqrt(c)), where c <= 2^31-1
Space: O(1)
'''
class Solution(object):
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        l = 0
        r = int(sqrt(c))
        while l <= r:
            cur = l * l + r * r
            if cur == c: return True
            if cur < c:
                l += 1
            else:
                r -= 1
        return False

# Day 11
# 1855. Maximum Distance Between a Pair of Values
'''
Solution 1
Iterate on input array B
Time O(n + m)
Space O(1)
'''
class Solution(object):
    def maxDistance(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        res = i = 0
        for j, b in enumerate(nums2):
            while i < len(nums1) and nums1[i] > b:
                i += 1
            if i == len(nums1): break
            res = max(res, j - i)
        return res
'''
Solution 2
Iterate on input array A
Time O(n + m)
Space O(1)
'''
class Solution(object):
    def maxDistance(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        res, j = 0, -1
        for i, a in enumerate(nums1):
            while j + 1 < len(nums2) and a <= nums2[j + 1]:
                j += 1
            res = max(res, j - i)
        return res

# 33. Search in Rotated Sorted Array
# Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)
        while l < r:
            m = l+(r-l) // 2
            if nums[m] == target:
                return m
            if nums[0] <= target < nums[m] or target < nums[m] < nums[0] or nums[m] < nums[0] <= target:
                r = m
            else:
                l = m+1

        return -1
# Day 12
# 153. Find Minimum in Rotated Sorted Array
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = 0
        r = len(nums) - 1
        while l < r:
            m = l + (r - l) // 2
            if nums[m] < nums[r]:
                # the mininum is in the left part
                r = m
            elif nums[m] > nums[r]:
                # the mininum is in the right part
                l = m + 1
        return nums[l]

# Binary Search II

# Day 1
# 209. Minimum Size Subarray Sum
'''Sliding Window
Intuition
Shortest Subarray with Sum at Least K
Actually I did this first, the same prolem but have negatives.
I suggest solving this prolem first then take 862 as a follow-up.

Explanation
The result is initialized as res = n + 1.
One pass, remove the value from sum s by doing s -= A[j].
If s <= 0, it means the total sum of A[i] + ... + A[j] >= sum that we want.
Then we update the res = min(res, j - i + 1)
Finally we return the result res

Complexity
Time O(N)
Space O(1)
'''
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        i, res = 0, len(nums) + 1
        for j in xrange(len(nums)):
            target -= nums[j]
            while target <= 0:
                res = min(res, j - i + 1)
                target += nums[i]
                i += 1
        return res % (len(nums) + 1)

# Binary Search
# O(n log n)
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        result = len(nums) + 1
        for idx, n in enumerate(nums[1:], 1):
            nums[idx] = nums[idx - 1] + n
        left = 0
        for right, n in enumerate(nums):
            if n >= target:
                left = self.binarySearch(left, right, nums, target, n)
                result = min(result, right - left + 1)
        return result if result <= len(nums) else 0

    def binarySearch(self, l, r, nums, target, n):
         while l < r:
            m = l+ (r - l) // 2
            if n - nums[m] >= target:
                l = m + 1
            else:
                r = m
        return l
# 611. Valid Triangle Number
#  Two-Pointer
'''
Complexity
Time: O(N^2), where N <= 1000 is number of elements in the array nums.
Space: O(logN), logN is the space complexity for sorting.
'''
class Solution(object):
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        n = len(nums)
        ans = 0
        if n < 3: return ans
        for k in range(2, n):
            l = 0
            r = k - 1
            while l < r:
                if nums[l] + nums[r] > nums[k]:
                    ans += r - l
                    r -= 1
                else:
                    l += 1
        return ans
# Binary Search TLE
class Solution(object):
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cnt = 0
        nums.sort()
        for i in range(len(nums)-2):
            k = i+2
            for j in range(i+1, len(nums)-1):
                if nums[i] == 0: break
                k = self.binarySearch(nums, 0, len(nums)-1, nums[i] + nums[j])
                cnt += k - j -1
        return cnt

    def binarySearch(self, nums, l, r, x):
        while l <= r:
            m = l+ (r - l) // 2
            if nums[m] >= x:
                r = m -1
            else:
                l = m +1
        return l
# Day 2
# 658. Find K Closest Elements
'''
Intuition
The array is sorted.
If we want find the one number closest to x,
we don't have to check one by one.
it's straightforward to use binary research.

Now we want the k closest,
the logic should be similar.


Explanation
Assume we are taking A[i] ~ A[i + k -1].
We can binary research i
We compare the distance between x - A[mid] and A[mid + k] - x

@vincent_gui listed the following cases:
Assume A[mid] ~ A[mid + k] is sliding window

case 1: x - A[mid] < A[mid + k] - x, need to move window go left
-------x----A[mid]-----------------A[mid + k]----------

case 2: x - A[mid] < A[mid + k] - x, need to move window go left again
-------A[mid]----x-----------------A[mid + k]----------

case 3: x - A[mid] > A[mid + k] - x, need to move window go right
-------A[mid]------------------x---A[mid + k]----------

case 4: x - A[mid] > A[mid + k] - x, need to move window go right
-------A[mid]---------------------A[mid + k]----x------

If x - A[mid] > A[mid + k] - x,
it means A[mid + 1] ~ A[mid + k] is better than A[mid] ~ A[mid + k - 1],
and we have mid smaller than the right i.
So assign left = mid + 1.

Important
Note that, you SHOULD NOT compare the absolute value of abs(x - A[mid]) and abs(A[mid + k] - x).
It fails at cases like A = [1,1,2,2,2,2,2,3,3], x = 3, k = 3

The problem is interesting and good.
Unfortunately the test cases is terrible.
The worst part of Leetcode test cases is that,
you submit a wrong solution but get accepted.

You didn't read my post and up-vote carefully,
then you miss this key point.


Complexity
Time O(log(N - K)) to binary research and find result
Space O(K) to create the returned list.
'''
class Solution(object):
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        l, r = 0, len(arr) - k
        while l < r:
            m = l + (r - l) / 2
            if x - arr[m] > arr[m + k] - x:
                l = m + 1
            else:
                r = m
        return arr[l:l + k]

# 1894. Find the Student that Will Replace the Chalk

class Solution(object):
    def chalkReplacer(self, chalk, k):
        """
        :type chalk: List[int]
        :type k: int
        :rtype: int
        """
        k %= sum(chalk)
        for i, a in enumerate(chalk):
            if k < a:
                return i
            k -= a
        return 0

class Solution(object):
    def chalkReplacer(self, chalk, k):
        """
        :type chalk: List[int]
        :type k: int
        :rtype: int
        """
        k %= sum(chalk)
        for i, c in enumerate(chalk):
            k -= c
            if k < 0:
                return i

# only works for python 3
# Time O(n)
# Space O(n)
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        return bisect.bisect(list(accumulate(chalk)), k % sum(chalk))

# Day 3
# 300. Longest Increasing Subsequence
# Binary Search / Patience sorting
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tails = [0] * len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) / 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size
'''
Solution 2: DP + Binary Search / Patience Sort

dp[i] := smallest tailing number of a increasing subsequence of length i + 1.

dp is an increasing array, we can use binary search to find the index to insert/update the array.

ans = len(dp)

Time complexity: O(nlogn)
Space complexity: O(n)
'''
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = []
        for x in nums:
            i = bisect_left(d, x)
            if i == len(d): 
                d.append(x)
            else:
                d[i] = x
        return len(d)

# Time Complexity: O(n^2) very slow
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums) == 0: return 0
        n = len(nums)
        f = [1] * n
        for i in range(1, n):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)
    
# 1760. Minimum Limit of Balls in a Bag
'''Binary Search
Explanation
Binary search the size of bag,
which is called penalty in this problem.

For each penalty value, we split the balls into bags with this value.
For example, the mid = 3,
A[i] = 2, we split it into [2], and operations = 0
A[i] = 3, we split it into [3], and operations = 0
A[i] = 4, we split it into [3,1], and operations = 1
A[i] = 5, we split it into [3,2], and operations = 1
A[i] = 6, we split it into [3,3], and operations = 1
A[i] = 7, we split it into [3,3,1], and operations = 2

The number of operation we need is (a - 1) / mid

If the total operation > max operations,
the size of bag is too small,
we set left = mid + 1

Otherwise,
this size of bag is big enough,
we set right = mid

We return the final result,
where result = left = right.

Complexity
Time O(nlog10^9)
Space O(1)'''
class Solution(object):
    def minimumSize(self, nums, maxOperations):
        """
        :type nums: List[int]
        :type maxOperations: int
        :rtype: int
        """
        l, r = 1, max(nums)
        while l < r:
            m = l + (r - l) / 2
            if sum((a - 1) / m for a in nums) > maxOperations:
                l = m + 1
            else:
                r = m
        return l

# Day 4
# 875. Koko Eating Bananas
'''
Solution: Binary Search
search for the smallest k [1, max_pile_height] such that eating time h <= H.

Time complexity: O(nlogh)
Space complexity: O(1)
'''
class Solution(object):
    def minEatingSpeed(self, piles, h):
        """
        :type piles: List[int]
        :type h: int
        :rtype: int
        """
        l = 1
        r = max(piles) + 1
        while l < r:
            m = (r - l) / 2 + l
            a = 0
            for p in piles:
                a += (p + m - 1) / m;
            if a <= h:
                r = m
            else:
                l = m + 1
        return l
# 1552. Magnetic Force Between Two Balls
'''
Explaination

We can use binary search to find the answer.
Define function count(d) that counts the number of balls can be placed in to baskets, under the condition that the minimum distance between any two balls is d.
We want to find the maximum d such that count(d) == m.

If the count(d) > m , we have too many balls placed, so d is too small.
If the count(d) < m , we don't have enough balls placed, so d is too large.
Since count(d) is monotonically decreasing with respect to d, we can use binary search to find the optimal d.


Complexity

Time complexity: O(Nlog(10^9)) or O(NlogM), where M = max(position) - min(position)
Space complexity: O(1)
'''
class Solution(object):
    def maxDistance(self, position, m):
        """
        :type position: List[int]
        :type m: int
        :rtype: int
        """
        n = len(position)
        position.sort()
        
        def count(d):
            ans, curr = 1, position[0]
            for i in range(1, n):
                if position[i] - curr >= d:
                    ans += 1
                    curr = position[i]
            return ans
        
        l, r = 0, position[-1] - position[0]
        while l <= r:
            mid = l + (r - l) // 2
            if count(mid) >= m:
                l = mid + 1
            else:
                r = mid - 1
        return l - 1
'''
Solution: Binary Search
Find the max distance that we can put m balls.

Time complexity: O(n*log(distance))
Space complexity: O(1)
'''
class Solution(object):
    def maxDistance(self, position, m):
        """
        :type position: List[int]
        :type m: int
        :rtype: int
        """
        position.sort()
        def getCount(d ):
            last, count = position[0], 1
            for x in position:
                if x - last >= d:
                    last = x
                    count += 1
            return count
        l, r = 0, position[-1] - position[0] + 1
        t = r
        while l < r:
            mid = l + (r - l) // 2
            if getCount(t - mid) >= m:
                r = mid
            else:
                l = mid + 1
        return t - l
# Day 5
# 287. Find the Duplicate Number
'''Explanation
Binary search the result.
If the sum > threshold, the divisor is too small.
If the sum <= threshold, the divisor is big enough.

Complexity
Time O(NlogM), where M = max(A)
Space O(1)'''
class Solution(object):
    def smallestDivisor(self, nums, threshold):
        """
        :type nums: List[int]
        :type threshold: int
        :rtype: int
        """
        l, r = 1, max(nums)
        while l < r:
            m = (l + r) / 2
            if sum((i + m - 1) / m for i in nums) > threshold:
                l = m + 1
            else:
                r = m
        return l

# 1283. Find the Smallest Divisor Given a Threshold
'''Approach : Binary Search

Complexity Analysis
Time Complexity: O(nlogn)
The outer loop uses binary search to identify a candidate - this runs in O(\log n)O(logn) time. 
For each candidate, we iterate over the entire array which takes O(n)O(n) time, 
resulting in a total of O(n \log n)O(nlogn) time.
Space Complexity: O(1)'''
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 'low' and 'high' represent the range of values of the target
        l = 1
        r = len(nums)
        
        while l < r:
            cur = l + (r - l) // 2
            count = 0

            # Count how many numbers are less than or equal to 'cur'
            count = sum(num <= cur for num in nums)
            if count > cur:
                duplicate = cur
                r = cur 
            else:
                l = cur + 1
                
        return duplicate

# Day 6
# 1898. Maximum Number of Removable Characters
'''
Idea
Binary Search k in interval [0, r), where r = len(removable)
For each mid, check if removable[:mid+1] could make p a subsequence of s.
If True, k could be larger, so we search in the right half; else, search in the left half.

Complexity
Time: O((p+s+r) * logr)
Space: O(r)
where r = len(removable); p = len(p); s = len(s).
'''
class Solution(object):
    def maximumRemovals(self, s, p, removable):
        """
        :type s: str
        :type p: str
        :type removable: List[int]
        :rtype: int
        """
        def check(m):
            i = j = 0
            remove = set(removable[:m+1])
            while i < len(s) and j < len(p):
                if i in remove:
                    i += 1
                    continue
                if s[i] == p[j]:
                    i += 1
                    j += 1
                else:
                    i += 1
            
            return j == len(p)
            
                
        # search interval is [l, r)
        l, r = 0, len(removable)+1
        
        while l < r:
            m = l + (r - l) // 2
            if check(m):
                l = m + 1
            else:
                r = m
                
        return l if l < len(removable) else l-1
# 1870. Minimum Speed to Arrive on Time
'''
y the condition, trains start only at the integer time. That means that if we pick any infinitely large speed, the smallest possible time to reach the office anyway will be bounded by that constrain. So we note that the minimum speed can be 1 km/h and the maximum speed that makes sense is max(dist) (at the first glance). So we have lower and upper bound for the speed and that means we can use Binary Search to find the speed that satisfies the condition.

As for the upper bound, consider the following example:

[1,1,100000]
2.01
The result for that unexpectedly will be 10000000 although we defined the upper bound as max(dist) which is for the example is just 100000. So when deriving the upper bound we should take into account that the last distance is not rounded during the calculation of the required time. Thus we should update our formula for the upper bound to ceil( max(dist) / dec ), where dec is the decimal part of the input hour.

Complexity:
Time: O(N*logN) on each BS shot we have to calculate required time to reach the office given the speed
Space: O(1)
'''
class Solution(object):
    def minSpeedOnTime(self, dist, hour):
        """
        :type dist: List[int]
        :type hour: float
        :rtype: int
        """
        # helper returns required time to reach the office given a speed
        def getRequiredTime(speed):
            time = sum([ceil(d/speed) for d in dist[:-1]])
            time += dist[-1]/speed
			
            return time
        
        dec = hour % 1 or 1 # decimal part of the `hour`
        l, r = 1, ceil( max(dist) / dec ) # min and max speed
		
        res = -1
    
        while l <= r:
            m = l + (r - l) // 2
            
            time = getRequiredTime(m)
            if time == hour:
                return int(m)
            
            if time < hour:
                res = m
                r = m - 1
            else:
                l = m + 1
            
        return int(res)
# Day 7
# 1482. Minimum Number of Days to Make m Bouquets
'''Binary Search 
Intuition
If m * k > n, it impossible, so return -1.
Otherwise, it's possible, we can binary search the result.
left = 1 is the smallest days,
right = 1e9 is surely big enough to get m bouquests.
So we are going to binary search in range [left, right].


Explanation
Given mid days, we can know which flowers blooms.
Now the problem is, given an array of true and false,
find out how many adjacent true bouquest in total.

If bouq < m, mid is still small for m bouquest.
So we turn left = mid + 1

If bouq >= m, mid is big enough for m bouquest.
So we turn right = mid


Complexity
Time O(Nlog(maxA))
Space O(1)

Note that the result must be one A[i],
so actually we can sort A in O(NlogK),
Where K is the number of different values.
and then binary search the index of different values.

Though I don't thik worth doing that.'''
class Solution(object):
    def minDays(self, bloomDay, m, k):
        """
        :type bloomDay: List[int]
        :type m: int
        :type k: int
        :rtype: int
        """
        if m * k > len(bloomDay): return -1
        left, right = 1, max(bloomDay)
        while left < right:
            mid = (left + right) / 2
            flow = bouq = 0
            for a in bloomDay:
                flow = 0 if a > mid else flow + 1
                if flow >= k:
                    flow = 0
                    bouq += 1
                    if bouq == m: break
            if bouq == m:
                right = mid
            else:
                left = mid + 1
        return left

# 1818. Minimum Absolute Sum Difference
'''
Clone nums1 and sort it as sorted1;
Traverse nums2, for each number find its corresponding difference from nums1: diff = abs(nums1[i] - nums2[i]) and use binary search against sorted1 to locate the index, idx, at which the number has minimum difference from nums2[i], then compute minDiff = abs(sorted1[idx] - nums2[i]) or abs(sorted1[idx - 1] - nums2[i]); Find the max difference mx out of all diff - minDiff;
Use the sum of the corresponding difference to substract mx is the solution.

Analysis:
Time: O(nlogn), space: O(n), where n = nums1.length = nums2.length.
'''
class Solution(object):
    def minAbsoluteSumDiff(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        diff_sum, mx = 0, float('-inf')
        sorted1 = sorted(nums1)        
        for i, (n1, n2) in enumerate(zip(nums1, nums2)):
            diff = abs(n1 - n2)
            diff_sum += diff
            idx = bisect.bisect_left(sorted1, n2) 
            if idx < len(sorted1):
                mx = max(mx, diff - abs(sorted1[idx] - n2))
            if idx > 0:
                mx = max(mx, diff - abs(sorted1[idx - 1] - n2))
        return (diff_sum - mx) % (10 ** 9 + 7)
        
# Day 8
# 240. Search a 2D Matrix II
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        c, r = len(matrix[0]) - 1, 0
        while c >= 0 and r < len(matrix):
            if matrix[r][c] > target:
                c -= 1
            elif matrix[r][c] < target:
                r += 1
            else:
                return True
        return False
# 275. H-Index II
'''
The basic idea of this solution is to use binary search to find the minimum index such that
citations[index] >= length(citations) - index
After finding this index, the answer is length(citations) - index.

This logic is very similar to the C++ function lower_bound or upper_bound.

Complexities:
Time: O(log n)
Space: O(1)
'''
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not citations: return 0
        n = len(citations)
        l, r = 0, n
        while l < r:
            m = l + (r -l)//2
            if m + citations[m] >= n:
                r = m
            else:
                l = m + 1                
        return n - l

# Day 9
# 1838. Frequency of the Most Frequent Element
''' Sliding Window
Intuition
Sort the input array A
Sliding window prolem actually,
the key is to find out the valid condition:
k + sum >= size * max
which is
k + sum >= (j - i + 1) * A[j]


Explanation
For every new element A[j] to the sliding window,
Add it to the sum by sum += A[j].
Check if it'a valid window by
sum + k < (long)A[j] * (j - i + 1)

If not, removing A[i] from the window by
sum -= A[i] and i += 1.

Then update the res by res = max(res, j - i + 1).

I added solution 1 for clearly expain the process above.
Don't forget to handle the overflow cases by using long.


Complexity
Time O(sort)
Space O(sort)


Solution : Use if clause
Just save some lines and improve a little time.'''
class Solution(object):
    def maxFrequency(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        i = 0
        nums.sort()
        for j in range(len(nums)):
            k += nums[j]
            if k < nums[j] * (j - i + 1):
                k -= nums[i]
                i += 1
        return j - i + 1

# 540. Single Element in a Sorted Array
class Solution(object):
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l, r = 0, len(nums)-1
        while l < r:
            m = l+ (r-l)//2
            if (m % 2 == 1 and nums[m - 1] == nums[m]) or (m % 2 == 0 and nums[m] == nums[m + 1]):
                l = m + 1
            else:
                r = m
        return nums[l]
# Day 10
# 222. Count Complete Tree Nodes
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        
        left, depth = root, 0
        while left.left:
            left, depth = left.left, depth + 1

        begin, end = (1<<depth), (1<<(depth+1)) - 1
        if self.Path(root,end): return end
        
        while begin + 1 < end:
            mid = (begin + end)//2
            if self.Path(root, mid):
                begin = mid
            else:
                end = mid
        return begin
        
    def Path(self, root, num):
        for s in bin(num)[3:]:
            if s == "0": 
                root = root.left
            else:
                root = root.right
            if not root: return False
        return True
# 1712. Ways to Split Array Into Three Subarrays
'''
The approach is not too hard, but the implementation was tricky for me to get right.

First, we prepare the prefix sum array, so that we can compute subarray sums in O(1). Then, we move the boundary of the first subarray left to right. This is the first pointer - i.

For each point i, we find the minimum (j) and maximum (k) boundaries of the second subarray:

nums[j] >= 2 * nums[i]
nums[sz - 1] - nums[k] >= nums[k] - nums[i]
Note that in the code and examples below, k points to the element after the second array. In other words, it marks the start of the (shortest) third subarray. This makes the logic a bit simpler.

With these conditions, sum(0, i) <= sum(i + 1, j), and sum(i + 1, k - 1) < sum(k, n). Therefore, for a point i, we can build k - j subarrays satisfying the problem requirements.

Final thing is to realize that j and k will only move forward, which result in a linear-time solution.

The following picture demonstrate this approach for the [4,2,3,0,3,5,3,12] test case.
'''
class Solution(object):
    def waysToSplit(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        sz, res, j, k = len(nums), 0, 0, 0
        for i in range(1, sz):
            nums[i] += nums[i - 1]
        for i in range(sz - 2):
            while j <= i or (j < sz - 1 and nums[j] < nums[i] * 2):
                j += 1
            while k < j or (k < sz - 1 and nums[k] - nums[i] <= nums[-1] - nums[k]):
                k += 1
            res = (res + k - j) % 1000000007
        return res

class Solution:
    def waysToSplit(self, nums: List[int]) -> int:
        sz, res, j, k = len(nums), 0, 0, 0
        nums = list(accumulate(nums))
        for i in range(sz - 2):
            while j <= i or (j < sz - 1 and nums[j] < nums[i] * 2):
                j += 1
            while k < j or (k < sz - 1 and nums[k] - nums[i] <= nums[-1] - nums[k]):
                k += 1
            res = (res + k - j) % 1000000007
        return res

# Day 11
# 826. Most Profit Assigning Work
'''Solution 
zip difficulty and profit as jobs.
sort jobs and sort 'worker'.
Use 2 pointers. For each worker, find his maximum profit best he can make under his ability.

Because we have sorted jobs and worker,
we will go through two lists only once.
this will be only O(D + W).
O(DlogD + WlogW), as we sort jobs.'''
class Solution(object):
    def maxProfitAssignment(self, difficulty, profit, worker):
        """
        :type difficulty: List[int]
        :type profit: List[int]
        :type worker: List[int]
        :rtype: int
        """
        jobs = sorted(zip(difficulty, profit))
        res = i = best = 0
        for ability in sorted(worker):
            while i < len(jobs) and ability >= jobs[i][0]:
                best = max(jobs[i][1], best)
                i += 1
            res += best
        return res

# 436. Find Right Interval
# My solution from contest with minimal cleanup.
# For each end point search for the first start point that is equal or higher in a previously constructed ordered list of start points.
# If there is one then return its index. If not return -1:
class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        l = sorted((e[0], i) for i, e in enumerate(intervals))
        res = []
        for e in intervals:
            r = bisect.bisect_left(l, (e[-1],))
            res.append(l[r][1] if r < len(l) else -1)
        return res
'''
Let us look carefully at our statement: for each interval i we need to find interval j, whose start is bigger or equal to the end point of interval i. We can rephrase this:
Given end of interval i, we need to find such point among starts of interval, which goes immedietly after this end of iterval i. How we can find this point? We can sort our intervals by starts (and hopefully there is no equal starts by statement) and then for each end of interval find desired place. Let us go through exapmle:
[1,12], [2,9], [3,10], [13,14], [15,16], [16,17] (I already sorted it by starts):

Look for number 12 in begs = [1,2,3,13,15,16]. What place we need? It is 3, because 12 <13 and 12 > 3.
Look for number 9, again place is 3.
Look for number 10, place is 3.
Look for number 14, place is 4, because 13<14<15.
Look for number 16, what is place? In is 5, because begs[5] = 16. Exactly for this reason we use bisect_left, which will deal with these cases.
Look for number 17, what is place? it is 6, but it means it is bigger than any number in our begs, so we should return -1.
So, what we do:

Sort our intervals by starts, but also we need to keep they original numbers, so I sort triplets: [start, end, index].
Create array of starts, which I call begs.
Creaty out result, which filled with -1.
Iterate over ints and for every end j, use bisect_left. Check that found index t < len(begs) and if it is, update out[k] = ints[t][2]. Why we update in this way, because our intervals in sorter list have different order, so we need to obtain original index.
Complexity: time complexity is O(n log n): both for sort and for n binary searches. Space complexity is O(n).
'''
class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        ints = sorted([[j,k,i] for i,[j,k] in enumerate(intervals)])
        begs = [i for i,_,_ in ints]
        out = [-1]*len(begs)
        for i,j,k in ints:
            t = bisect.bisect_left(begs, j)
            if t < len(begs):
                out[k] = ints[t][2]
        
        return out
        
# Day 12
# 81. Search in Rotated Sorted Array II
# moves the right pointer one step forward when nums[mid] == nums[r],
# as we don't know target is in left part or in right part:
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        if not nums:
            return False
        l, r = 0, len(nums)-1
        while l < r:
            mid = l + (r-l)//2
            if nums[mid] == target:
                return True
            if nums[mid] < nums[r]:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            elif nums[mid] > nums[r]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                r -= 1
        return nums[l] == target

# 162. Find Peak Element
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return 0
        l, r = 0, len(nums) - 1
        while l < r:
            m = l +(r-l) // 2
            if nums[m] > nums[m+1]:  # Found peak
                r = m
            else:  # Find peak on the right
                l = m + 1
        
        return l

# Day 13
# 154. Find Minimum in Rotated Sorted Array II
'''
Binary Search
First, we take
low (lo) as 0
high (hi) as nums.length-1

By default, if nums[lo]<nums[hi] then we return nums[lo] because the array was never rotated, or is rotated n times

After entering while loop, we check
if nums[mid] > nums[hi] => lo = mid + 1 because the minimum element is in the right half of the array
else if nums[mid] < nums[hi] => hi = mid because the minimum element is in the left half of the array
else => hi-- dealing with duplicate values
then we return nums[hi]
'''
# contain duplicates
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + (r -l) // 2
            if nums[m] > nums[r]:
                l = m + 1
            else:
                r = m if nums[r] != nums[m] else r - 1
        return nums[l]
# also works...slow btw
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return min(nums)

# no duplicate => not works in this question
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + (r -l) / 2
            if nums[m] > nums[r]:
                l = m + 1
            else:
                r = m
        return nums[l] 

# 528. Random Pick with Weight
# Python solution
class Solution(object):

    def __init__(self, w):
        """
        :type w: List[int]
        """
        for i in range(1,len(w)):
            w[i] += w[i-1]
        self.w = w


    def pickIndex(self):
        """
        :rtype: int
        """
        l, r = 0, len(self.w)-1
        target = random.randint(1,self.w[-1])
        while l < r:
            mid = l + (r-l)//2
            if target <= self.w[mid]:
                r = mid
            else:
                l = mid+1
        return l
        
# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()

# Python 3 solution
class Solution:

    def __init__(self, w: List[int]):
        self.w = list(itertools.accumulate(w))

    def pickIndex(self) -> int:
        return bisect.bisect_left(self.w, random.randint(1, self.w[-1]))

# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()

# Day 14
# 1508. Range Sum of Sorted Subarray Sums
'''
Solution : Priority Queue / Min Heap
For each subarray, start with one element e.g nums[i], put them into a priority queue (min heap). Each time, we have the smallest subarray sum, and extend that subarray and put the new sum back into priority queue. Thought it has the same time complexity as the brute force one in worst case, but space complexity can be reduce to O(n).

Time complexity: O(n^2logn)
Space complexity: O(n)
'''
class Solution(object):
    def rangeSum(self, nums, n, left, right):
        """
        :type nums: List[int]
        :type n: int
        :type left: int
        :type right: int
        :rtype: int
        """
        q = [(num, i) for i, num in enumerate(nums)]
        heapq.heapify(q)
        ans = 0
        for k in range(1, right + 1):
            s, i = heapq.heappop(q)
            if k >= left:
                ans += s
            if i + 1 < n:
                heapq.heappush(q, (s + nums[i + 1], i + 1))
        return ans % int(1e9 + 7)

'''Binary Search
Explanation
count_sum_under counts the number of subarray sums that <= score
sum_k_sums returns the sum of k smallest sums of sorted subarray sums.
kth_score returns the kth sum in sorted subarray sums.

Oral explanation refers to youtube channel.

Complexity
Time O(NlogSum(A))
Space O(N)
'''
# it's not pass all the cases... There's stackoverflow
def rangeSum(self, A, n, left, right):
        # B: partial sum of A
        # C: partial sum of B
        # Use prefix sum to precompute B and C
        B, C = [0] * (n + 1), [0] * (n + 1)
        for i in range(n):
            B[i + 1] = B[i] + A[i]
            C[i + 1] = C[i] + B[i + 1]

        # Use two pointer to
        # calculate the total number of cases if B[j] - B[i] <= score
        def count_sum_under(score):
            res = i = 0
            for j in range(n + 1):
                while B[j] - B[i] > score:
                    i += 1
                res += j - i
            return res

        # calculate the sum for all numbers whose indices are <= index k
        def sum_k_sums(k):
            score = kth_score(k)
            res = i = 0
            for j in range(n + 1):
                # Proceed until B[i] and B[j] are within score
                while B[j] - B[i] > score:
                    i += 1
                res += B[j] * (j - i + 1) - (C[j] - (C[i - 1] if i else 0))
            return res - (count_sum_under(score) - k) * score

        # use bisearch to find how many numbers ae below k
        def kth_score(k):
            l, r = 0, B[n]
            while l < r:
                m = (l + r) / 2
                if count_sum_under(m) < k:
                    l = m + 1
                else:
                    r = m
            return l

        # result between left and right can be converted to [0, right] - [0, left-1] (result below right - result below left-1)
        return sum_k_sums(right) - sum_k_sums(left - 1)

# 1574. Shortest Subarray to be Removed to Make Array Sorted
'''
Solution: Two Pointers
Find the right most j such that arr[j – 1] > arr[j], 
if not found which means the entire array is sorted return 0. 
Then we have a non-descending subarray arr[j~n-1].

We maintain two pointers i, j, such that arr[0~i] is non-descending and arr[i] <= arr[j] 
which means we can remove arr[i+1~j-1] to get a non-descending array. 
Number of elements to remove is j - i - 1 .

Time complexity: O(n)
Space complexity: O(1)

Since we can only remove a subarray, the final remaining elements must be either: (1) solely a prefix, (2) solely a suffix or (3) a merge of the prefix and suffix.

Find the monotone non-decreasing prefix [a_0 <= ... a_i | ...]
l is the index such that arr[l+1] < arr[l]
Find the monotone non-decreasing suffix [... | a_j <= ... a_n]
r is the index such that arr[r-1] > arr[r]
Try to "merge 2 sorted arrays", if we can merge, update our minimum to remove.
'''
class Solution(object):
    def findLengthOfShortestSubarray(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        l, r = 0, len(arr) - 1
        while l < r and arr[l+1] >= arr[l]:
            l += 1
        if l == len(arr) - 1:
            return 0 # whole array is sorted
        while r > 0 and arr[r-1] <= arr[r]:
            r -= 1
        toRemove = min(len(arr) - l - 1, r) # case (1) and (2)
		
		# case (3): try to merge
        for iL in range(l+1):
            if arr[iL] <= arr[r]:
                toRemove = min(toRemove, r - iL - 1)
            elif r < len(arr) - 1:
                r += 1
            else:
                break
        return toRemove

# Day 15
# 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
# Prefix Sum
class Solution(object):
    def maxSideLength(self, mat, threshold):
        """
        :type mat: List[List[int]]
        :type threshold: int
        :rtype: int
        """
        # GET DIMENSIONS
        nrows, ncols = len(mat), len(mat[0])

        # SETUP THE PREFIX SUM MATRIX
        prefix = [[0 for _ in range(ncols + 1)] for _ in range(nrows + 1)]
        
        # FILL THE CELLS - TOP RECT + LEFT RECT - TOP LEFT DOUBLE-COUNTED RECT
        for i in range(nrows):
            for j in range(ncols):
                prefix[i + 1][j + 1] = prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j] + mat[i][j]
        
        # for row in prefix:
        #     print(row)
            
        '''
        1. INITIALIZE MAX_SIDE = 0
        2. AT EACH CELL, WE'LL CHECK IF RECTANGLE (OR SQUARE) FROM [I - MAX, J - MAX] TO [I, J], BOTH INCLUSIVE, IS <= THRESHOLD
        '''
        
        # INITIALIZE MAX SIDE
        max_side = 0
        
        # CHECK IF RECTANGLE (OR SQUARE) FROM [I - MAX, J - MAX] TO [I, J] <= THRESHOLD
        for i in range(nrows):
            for j in range(ncols): 
                
                # CHECK IF WE CAN SUBTRACT MAX_SIDE
                if min(i, j) >= max_side:
                    curr = prefix[i + 1][j + 1]
                    top = prefix[i - max_side][j + 1]
                    left = prefix[i + 1][j - max_side]
                    topLeft = prefix[i - max_side][j - max_side]
                    total = curr - top - left + topLeft
                    
                    # print(f"CURR : {curr} | TOP : {top} | LEFT : {left} | TOP_LEFT : {topLeft}")
                    # print(f"TOTAL : {total}\n")
                    
                    # UPDATE MAX_SIDE IFF TOTAL <= THRESHOLD
                    if total <= threshold:
                        max_side += 1
                
        # RETURN MAX SIDE
        return max_side

# 1498. Number of Subsequences That Satisfy the Given Sum Condition
'''  Two Sum
Intuition
Almost same as problem two sum.
If we want to know the count of subarray in sorted array A,
then it's exactly the same.
Make sure you can do two sum before continue.


Explanation
Sort input A first,
For each A[i], find out the maximum A[j]
that A[i] + A[j] <= target.

For each elements in the subarray A[i+1] ~ A[j],
we can pick or not pick,
so there are 2 ^ (j - i) subsequences in total.
So we can update res = (res + 2 ^ (j - i)) % mod.

We don't care the original elements order,
we only want to know the count of sub sequence.
So we can sort the original A, and the result won't change.


Complexity
Time O(NlogN)
Space O(1) for python
(O(N) space for java and c++ can be save anyway)'''
class Solution(object):
    def numSubseq(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        l, r = 0, len(nums) - 1
        res = 0
        mod = 10**9 + 7
        while l <= r:
            if nums[l] + nums[r] > target:
                r -= 1
            else:
                res += pow(2, r - l, mod)
                l += 1
        return res % mod

# Day 16
# 981. Time Based Key-Value Store
class TimeMap(object):

    def __init__(self):
        self.dic = collections.defaultdict(list)

    def set(self, key, value, timestamp):
        """
        :type key: str
        :type value: str
        :type timestamp: int
        :rtype: None
        """
        self.dic[key].append([timestamp, value])

    def get(self, key, timestamp):
        """
        :type key: str
        :type timestamp: int
        :rtype: str
        """
        arr = self.dic[key]
        n = len(arr)
        
        left = 0
        right = n
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid][0] <= timestamp:
                left = mid + 1
            elif arr[mid][0] > timestamp:
                right = mid
        
        return "" if right == 0 else arr[right - 1][1]

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

# 1300. Sum of Mutated Array Closest to Target
'''
Explanation
Binary search is O(NlogMax(A)).
In order to ruduce the difficulty, it constrains A[i] < 10 ^ 5.

In this solution,
we sort the input and compared A[i] with target one by one.

Sort the array A in decreasing order.
We try to make all values in A to be the min(A) (the last element)
If target >= min(A) * n, we doesn't hit our target yet.
We should continue to try a value bigger.
So we pop the min(A) value.
Consider that it won't be affected anymore,
we can remove it from target by target -= A.pop()
We continue doing step 2-4, until the next number is too big for target.
We split the the target evenly, depending on the number of element left in A
At this point, @bobalice help explain the round part:
if A is empty means its impossible to reach target so we just return maximum element.
If A is not empty, intuitively the answer should be the nearest integer to target / len(A).

Since we need to return the minimum such integer if there is a tie,
if target / len(A) has 0.5 we should round down,


Complexity
Time O(NlogN)
Space O(1)
'''
class Solution(object):
    def findBestValue(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        arr.sort(reverse=1)
        maxA = arr[0]
        while arr and target >= arr[-1] * len(arr):
            target -= arr.pop()
        return int(round((target - 0.0001) / len(arr))) if arr else maxA

# Day 17
# 1802. Maximum Value at a Given Index in a Bounded Array
''' Binary Search
Explanation
We first do maxSum -= n,
then all elements needs only to valid A[i] >= 0

We binary search the final result between left and right,
where left = 0 and right = maxSum.

For each test, we check minimum sum if A[index] = a.
The minimum case would be A[index] is a peak in A.
It's arithmetic sequence on the left of A[index] with difference is 1.
It's also arithmetic sequence on the right of A[index] with difference is -1.

On the left, A[0] = max(a - index, 0),
On the right, A[n - 1] = max(a - ((n - 1) - index), 0),

The sum of arithmetic sequence {b, b+1, ....a},
equals to (a + b) * (a - b + 1) / 2.


Complexity
Because O(test) is O(1)
Time O(log(maxSum))
Space O(1)
'''
class Solution(object):
    def maxValue(self, n, index, maxSum):
        """
        :type n: int
        :type index: int
        :type maxSum: int
        :rtype: int
        """
        def test(a):
            b = max(a - index, 0)
            res = (a + b) * (a - b + 1) / 2
            b = max(a - ((n - 1) - index), 0)
            res += (a + b) * (a - b + 1) / 2
            return res - a

        maxSum -= n
        left, right = 0, maxSum
        while left < right:
            mid = (left + right + 1) / 2
            if test(mid) <= maxSum:
                left = mid
            else:
                right = mid - 1
        return left + 1

# 1901. Find a Peak Element II
class Solution(object):
    def findPeakGrid(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[int]
        """
        top = 0
        bottom = len(mat)-1
        while bottom > top:
            mid = (top + bottom) // 2
            if max(mat[mid]) > max(mat[mid+1]):
                bottom = mid
            else:
                top = mid+1
        return [bottom,mat[bottom].index(max(mat[bottom]))]

# Day 18
# 1146. Snapshot Array
''' Binary Search
Intuition
Instead of copy the whole array,
we can only record the changes of set.


Explanation
The idea is, the whole array can be large,
and we may take the snap tons of times.
(Like you may always ctrl + S twice)

Instead of record the history of the whole array,
we will record the history of each cell.
And this is the minimum space that we need to record all information.

For each A[i], we will record its history.
With a snap_id and a its value.

When we want to get the value in history, just binary search the time point.


Complexity
Time O(logS)
Space O(S)
where S is the number of set called.

SnapshotArray(int length) is O(N) time
set(int index, int val) is O(1) in Python and O(logSnap) in Java
snap() is O(1)
get(int index, int snap_id) is O(logSnap)
'''
class SnapshotArray(object):

    def __init__(self, length):
        """
        :type length: int
        """
        self.A = [[[-1, 0]] for _ in xrange(length)]
        self.snap_id = 0

    def set(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        self.A[index].append([self.snap_id, val])

    def snap(self):
        """
        :rtype: int
        """
        self.snap_id += 1
        return self.snap_id - 1
        

    def get(self, index, snap_id):
        """
        :type index: int
        :type snap_id: int
        :rtype: int
        """
        i = bisect.bisect(self.A[index], [snap_id + 1]) - 1
        return self.A[index][i][1]


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)

# 1488. Avoid Flood in The City
class Solution(object):
    def avoidFlood(self, rains):
        """
        :type rains: List[int]
        :rtype: List[int]
        """
        dic = collections.defaultdict(list)
        ret = [-1] * len(rains)
        to_empty = [] # index
        
        for day,lake in enumerate(rains):
            dic[lake].append(day)
        
        for i in range(len(rains)):
            lake = rains[i]
            if lake:
                if dic[lake] and dic[lake][0] < i:
                    return []
                if dic[lake] and len(dic[lake])>1:
                    heapq.heappush(to_empty,dic[lake][1])
            else:
                if to_empty:
                    ret[i] = rains[heapq.heappop(to_empty)]
                    dic[ret[i]].pop(0)
                else:
                    ret[i] = 1
        return ret

# Day 19
# 1562. Find Latest Group of Size M
'''
Solution: Hashtable
Similar to LC 128

Time complexity: O(n)
Space complexity: O(n)
'''
class Solution(object):
    def findLatestStep(self, arr, m):
        """
        :type arr: List[int]
        :type m: int
        :rtype: int
        """
        if m == len(arr): return m
        length = [0] * (len(arr) + 2)
        res = -1
        for i, a in enumerate(arr):
            left, right = length[a - 1], length[a + 1]
            if left == m or right == m:
                res = i
            length[a - left] = length[a + right] = left + right + 1
        return res

# 1648. Sell Diminishing-Valued Colored Balls
'''binary serach solution
First, it should be clear that we want to sell from the most abundant balls as much as possible as it is valued more than less abundant balls. In the spirit of this, we propose the below algo

sort inventory in reverse order and append 0 at the end (for termination);
scan through the inventory, and add the difference between adjacent categories to answer.
Assume inventory = [2,8,4,10,6]. Then, we traverse inventory following 10->8->6->4->2->0. The evolution of inventory becomes

10 | 8 | 6 | 4 | 2
 8 | 8 | 6 | 4 | 2
 6 | 6 | 6 | 4 | 2
 4 | 4 | 4 | 4 | 2
 2 | 2 | 2 | 2 | 2 
 0 | 0 | 0 | 0 | 0 
Of course, if in any step, we have enough order, return the answer.
'''
class Solution(object):
    def maxProfit(self, inventory, orders):
        """
        :type inventory: List[int]
        :type orders: int
        :rtype: int
        """
        fn = lambda x: sum(max(0, xx - x) for xx in inventory) # balls sold 
    
        # last true binary search 
        lo, hi = 0, 10**9
        while lo < hi: 
            mid = lo + hi + 1 >> 1
            if fn(mid) >= orders: lo = mid
            else: hi = mid - 1
        
        ans = sum((x + lo + 1)*(x - lo)//2 for x in inventory if x > lo)
        return (ans - (fn(lo) - orders) * (lo + 1)) % 1000000007

# Day 20
# 1201. Ugly Number III
# python 3 only
class Solution:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        def enough(num) -> bool:
            total = mid//a + mid//b + mid//c - mid//ab - mid//ac - mid//bc + mid//abc
            return total >= n

        ab = a * b // math.gcd(a, b)
        ac = a * c // math.gcd(a, c)
        bc = b * c // math.gcd(b, c)
        abc = a * bc // math.gcd(a, bc)
        left, right = 1, 10 ** 10
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left

# Python Binary Search 
# https://leetcode.com/problems/ugly-number-iii/discuss/769707/Python-Clear-explanation-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems.
class Solution(object):
    def nthUglyNumber(self, n, a, b, c):
        """
        :type n: int
        :type a: int
        :type b: int
        :type c: int
        :rtype: int
        """
        def enough(num):
            total = mid//a + mid//b + mid//c - mid//ab - mid//ac - mid//bc + mid//abc
            return total >= n
        def gcd(a, b):
            if (a == 0): return b
            return gcd(b % a, a)
    
        ab = a * b // gcd(a, b)
        ac = a * c // gcd(a, c)
        bc = b * c // gcd(b, c)
        abc = a * bc // gcd(a, bc)
        left, right = 1, 10 ** 10
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left

# Python Binary Search 
'''
The term "ugly number" seems to reflect a poorly-defined concept. 
Upon Googling it, I can only find it in a few places such as LC, GFG, etc. 
Even in the few posts on LC, the concept varies. For example, in 263. 
Ugly Number, an ugly number is a positive integer whose only factors are 2, 3 
and 5, but 1 is treated as an ugly number. 
This definition is consistent with that of 264. Ugly Number II. 
But in 1201. Ugly Number III, ugly number becomes positive integers divisible by given factors 
(let's still use 2, 3, 5 unless stated otherwise), and 1 is not considered ugly any more.

Let's refer to the definition in 263 and 264 "Def 1" and the definition in 1201 "Def 2". 
Under Def 1, the first few ugly numbers are 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, ... 
while under Def 2 the first few ugly numbers are 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, ... . 
The similarity is obvious at first glance. 
But if you look deeper, a fundamental difference can be revealed. 
Namely, under Def 1, ugly number is self-generated, i.e. 
large ugly numbers are generated by multiplying factors with small ugly numbers. 
Because of this, ugly numbers become rarer as number becomes larger. 
However, under Def 2, ugly numbers are periodic. 
The pattern repeats when least common multiple is reached.

To reflect the "self-generating" property of ugly number under Def 1, 
263 and 264 can be solved using dynamic programming. 
For example, this post and this post implement the solution using top-down approach. 
But 1201 needs to be solved in a completely different way. 
In the spirit of this difference, I think it is more confusing than helpful to put 1201 
in the ugly number series. 
It is probably clearer if this is treated as a completely independent problem.
'''
class Solution(object):
    def nthUglyNumber(self, n, a, b, c):
        """
        :type n: int
        :type a: int
        :type b: int
        :type c: int
        :rtype: int
        """
        def gcd(a, b):
            if (a == 0): return b
            return gcd(b % a, a)
    
        # inclusion-exclusion principle
        ab = a*b//gcd(a, b)
        bc = b*c//gcd(b, c)
        ca = c*a//gcd(c, a)
        abc = ab*c//gcd(ab, c)
        
        lo, hi = 1, n*min(a, b, c)
        while lo < hi: 
            mid = lo + hi >> 1
            if mid//a + mid//b + mid//c - mid//ab - mid//bc - mid//ca + mid//abc < n: lo = mid + 1
            else: hi = mid 
        return lo 
# 911. Online Election
'''
Initialization part
In the order of time, we count the number of votes for each person.
Also, we update the current lead of votes for each time point.
if (count[person] >= count[lead]) lead = person

Time Complexity: O(N)


Query part
Binary search t in times,
find out the latest time point no later than t.
Return the lead of votes at that time point.

Time Complexity: O(logN)
'''

class TopVotedCandidate(object):

    def __init__(self, persons, times):
        """
        :type persons: List[int]
        :type times: List[int]
        """
        self.leads, self.times, count = [], times, {}
        lead = -1
        for p in persons:
            count[p] = count.get(p, 0) + 1
            if count[p] >= count.get(lead, 0): lead = p
            self.leads.append(lead)

    def q(self, t):
        """
        :type t: int
        :rtype: int
        """
        return self.leads[bisect.bisect(self.times, t) - 1]


# Your TopVotedCandidate object will be instantiated and called as such:
# obj = TopVotedCandidate(persons, times)
# param_1 = obj.q(t)