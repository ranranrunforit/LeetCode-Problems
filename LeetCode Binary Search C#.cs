// Day 1
// 704. Binary Search
// Solution: Binary Search
// Time complexity: O(logn)
// Space complexity: O(1)
public class Solution {
    public int Search(int[] nums, int target) {
        int l = 0, r = nums.Length;
        while (l<r)
        {
            int m = l+(r-l)/2;
            if (nums[m] == target)
            {
                return m;
            }
            else if(nums[m] > target)
            {
                r = m;
            }
            else
            {
                l = m+1;
            }
        }
        return -1;
    }
}

// 374. Guess Number Higher or Lower

/** 
 * Forward declaration of guess API.
 * @param  num   your guess
 * @return 	     -1 if num is higher than the picked number
 *			      1 if num is lower than the picked number
 *               otherwise return 0
 * int guess(int num);
 */

public class Solution : GuessGame {
    public int GuessNumber(int n) {
        int l = 0, r = n;
        while (l<r)
        {
            int m = l+(r-l)/2;
            if (guess(m) == 0)
            {
                return m;
            }
            else if(guess(m) == -1)
            {
                r = m;
            }
            else
            {
                l = m+1;
            }
        }
        return l;
    }
}

// Day 2
// 35. Search Insert Position
public class Solution {
    public int SearchInsert(int[] nums, int target) {
        int l = 0, r = nums.Length;
        while(l<r)
        {
            int m = l+(r-l)/2;
            if(nums[m]==target)
            {
                return m;
            }
            else if(nums[m]>target){
                r = m;
            }
            else{
                l = m+1;
            }
        }
        return l;  // or return r would work too
    
    }
}

// 852. Peak Index in a Mountain Array
// Find the smallest l such that A[l] > A[l + 1].
// Time complexity: O(logn)
// Space complexity: O(1)
public class Solution {
    public int PeakIndexInMountainArray(int[] arr) {
        int l = 0, r = arr.Length;
        while(l<r)
        {
            int m = l+(r-l)/2;
            if (arr[m] > arr[m+1])
            {
                r = m;
            }
            else
            {
                l = m+1;
            }
        }
        return l;
    }
}

// Day 3 
// 367. Valid Perfect Square
// Binary Search
// Complexity:
// Time: O(logNum)
// Space: O(1)
// without Math.Pow
// using long avoid overflow for test case 2147483647
public class Solution {
    public bool IsPerfectSquare(int num) {
        if(num == 1) return true;
        long l = 0, r = num;
        while(l < r) // r could be 1
        {
            long m = l+(r-l)/2; //long to avoid overflow incase (left+right)>2147483647
            
            if(m * m == num)//(Math.Pow(m,2)==num) works for long type
            {// check if mid is perfect square
                return true;
            }
            else if( m * m > num)//(Math.Pow(m,2)>num) works
            {// mid is large -> to left to decrease mid
                r = m;
            }
            else
            {// mid is small -> go right to increase mid
                l = m + 1;
            }
        }
        return false;
    }
}
// Binary Search
// using int 
// without Math.Pow
public class Solution {
    public bool IsPerfectSquare(int num) {
        int l = 1, r = num; // l have to be 1 if otherwise case when r = 1, m = 0, Divide Error
        while (l <= r) {
            int m = l + (r - l) / 2; // to avoid overflow incase (left+right)>2147483647
            int res = num / m, remain = num % m;
            if (res == m && remain == 0) {return true; }// check if mid * mid == num
            if (res <= m) { 
                 r = m - 1; // mid is large -> to left to decrease mid
            } 
            else {// mid is small -> go right to increase mid
                l = m + 1;
               
            }
        }
        return false;
    }
}
// Binary Search
//mine
public class Solution {
    public bool IsPerfectSquare(int num) {
        if (num == 1) return true;
        int l = 0, r = num;
        while (l < r) {
            int m = l + (r - l) / 2; // to avoid overflow incase (left+right)>2147483647
            int res = num / m, remain = num % m;
            if (res == m && remain == 0) {return true; }// check if mid * mid == num
            if (res <= m) { // res < m works too!
                 r = m; // mid is large -> to left to decrease mid
            } 
            else {// mid is small -> go right to increase mid
                l = m + 1;
               
            }
        }
        return false;
    }
}
// Binary Search
// with Math.Pow
// mine
public class Solution {
    public bool IsPerfectSquare(int num) {
        if(num==1)
        {
            return true;
        }
        int l=0,r=num;
        while(l<r)
        {
            int m = l+(r-l)/2;
            if(Math.Pow(m,2)==num)
            {
                return true;
            }
            else if(Math.Pow(m,2)>num)
            {
                r = m;
            }
            else
            {
                l = m+1;
            }
        }
        return false;
    }
}
// a slightly better option maybe
public class Solution {
    public bool IsPerfectSquare(int num) {
        int l=0,r=num;
        while(l<=r)
        {
            int m = l+(r-l)/2;
            if(Math.Pow(m,2)==num)
            {
                return true;
            }
            else if(Math.Pow(m,2)>num)
            {
                r = m-1;
            }
            else
            {
                l = m+1;
            }
        }
        return false;
    }
}
// 1385. Find the Distance Value Between Two Arrays
// Binary Search
public class Solution {
    public int FindTheDistanceValue(int[] arr1, int[] arr2, int d) {
        Array.Sort(arr2);

        int count = 0;
        for(int i=0;i<arr1.Length;i++)
        {
            if (Check(arr1[i],arr2,d)==true )
            {
                count++;
            }
        }
        return count;

    }
    
    public bool Check(int val, int[] arr2, int d){
        int l = 0, r = arr2.Length;
        while(l<r)
        {
            int m = l+(r-l)/2;
            if(Math.Abs(val-arr2[m]) <= d)
            {
                return false;
            }
            else if(val < arr2[m]) // && Math.Abs(val-arr2[m]) > d
            {
                r = m;
            }
            else // val > arr2[m] && Math.Abs(val-arr2[m]) > d
            {
                l = m + 1;
            }
        }
        return true;
    }
}

// Day 4
// 69. Sqrt(x)
// Solution : Newton’s method
public class Solution {
    public int MySqrt(int x) {
        if (x == 0) return 0;
        long a = (long) x;
        while (a > x / a)
          a = (a + x / a) / 2;
        return unchecked((int)a);
        
    }
}

// solution 1:
public class Solution {
    public int MySqrt(int x) {
        /* Look for the critical point: i * i <= x && (i+1)(i+1) > x
        A little trick is using i <= x / i for comparison,
        instead of i * i <= x, to avoid exceeding integer upper limit. */
        if (x==0) return 0;
        int l = 1;
        int r = x;
        while (l < r) {
          int m = l + (r - l) / 2;
          if (x / m >= m && x / (m + 1) < (m + 1) ){
            return m ;
          } 
          else if (x / m < m) {
            r = m ;
          } else {
            l = m + 1;
          }
        }
        return l;
    }
}

// solution 2:
public class Solution {
    public int MySqrt(int x) {
        int l = 1;
        int r = x;
        while (l <= r) {
          int m = l + (r - l) / 2;
          if (m > x / m) {
            r = m - 1;
          } else {
            l = m + 1;
          }
        }
        return r;
    }
}

// 744. Find Smallest Letter Greater Than Target
public class Solution {
    public char NextGreatestLetter(char[] letters, char target) {
        int l = 0, r = letters.Length;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (letters[m] <= target) l = m + 1;
            else r = m;
        }
        // if our insertion position says to insert target into the last position letters.length,
        // we return letters[0] instead. 
        //This is what the modulo operation does.
        return letters[l % letters.Length];
    }
}

// Day 5
// 278. First Bad Version
/* The isBadVersion API is defined in the parent class VersionControl.
      bool IsBadVersion(int version); */

public class Solution : VersionControl {
    public int FirstBadVersion(int n) {
        int l = 0, r = n;
        while (l<r)
        {
            int m = l+(r-l)/2;
            if (IsBadVersion(m) == true)
            {
                r = m;
            }
            else if(IsBadVersion(m) == false)
            {
                l = m+1;
            }

        }
        return l;
    }
}
// 34. Find First and Last Position of Element in Sorted Array
/*
Solution: Binary Search
Basically this problem asks you to implement lower_bound and upper_bound using binary search.

Time complexity: O(logn)
Space complexity: O(1)
*/
public class Solution {
    public int[] SearchRange(int[] nums, int target) {
        return new int[] {firstPos(nums, target), lastPos(nums, target)};
  }

  public int firstPos(int[] nums, int target) {
    int l = 0, r = nums.Length;
    while (l < r) {
      int m = l + (r - l) / 2;      
      if (nums[m] >= target) { //have to have = here so that it find the first element
        r = m;
      } else {
        l = m + 1;
      }
    }
    
    if (l == nums.Length || nums[l] != target) return -1;
    return l;    
  }
 
  public int lastPos(int[] nums, int target) {
    int l = 0, r = nums.Length;
    while (l < r) {
      int m = l + (r - l) / 2;      
      if (nums[m] > target) { // if use = here then might miss other duplicate
        r = m; 
      } else {
        l = m + 1;
      }
    }
    // l points to the first element that greater than target.
    --l;        
    if (l < 0 || nums[l] != target) return -1;
    return l;
  }
}

// Day 6
// 441. Arranging Coins
public class Solution {
    public int ArrangeCoins(int n) {
        long l = 0, r = n;
        long m, cur;
        while (l <= r) {
          m = l + (r - l) / 2;
          cur = m * (m + 1) / 2;

          if (n == cur) return (int)m;

          if (n < cur) {
            r = m - 1;
          } else {
            l = m + 1;
          }
        }
        return (int)r;
    }
}

// 1539. Kth Missing Positive Number
/*
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
*/
public class Solution {
    public int FindKthPositive(int[] arr, int k) {
         int l = 0, r = arr.Length, m;
            while (l < r) {
                m = l + (r - l) / 2;
                if (arr[m] - 1 - m < k)
                    l = m + 1;
                else
                    r = m;
            }
            return l + k;
    }
}

// Day 7
// 167. Two Sum II - Input Array Is Sorted
// two-pointer
public class Solution {
    public int[] TwoSum(int[] numbers, int target) {
        int l = 0,  r = numbers.Length-1;
        while(l<r){
            if(numbers[l]+numbers[r]==target)
            {
                return new int[]{l+1,r+1};
            }
            else if(numbers[l]+numbers[r]>target){
                r--;
            }
            else
            {
                l++;
            }
        }
        return new int[]{};
    }
}

// Binary Search
public class Solution {
    public int[] TwoSum(int[] numbers, int target) {
        for(int i = 0; i<numbers.Length; i++){
            int l = i+1, r = numbers.Length, tmp = target - numbers[i];
            while(l<r){
                int m = l +(r-l)/2;
                if(numbers[m] == tmp)
                {
                    return new int[]{i+1,m+1};
                }
                else if(numbers[m] < tmp){
                    l = m + 1;
                }
                else
                {
                    r = m;
                }
            }
        }
        return new int[]{};
    }
}

// 1608. Special Array With X Elements Greater Than or Equal X
/*
Concept is similar to H-index
After while loop, we can get i which indicates there are already i items larger or equal to i.
Meanwhile, if we found i == nums[i], there will be i + 1 items larger or equal to i, which makes array not special.
Time: O(sort), can achieve O(N) if we use counting sort
Space: O(sort)
*/
public class Solution {
    public int SpecialArray(int[] nums) {
        nums = nums.OrderByDescending(c => c).ToArray(); 
        // Array.Sort(nums, (x, y) => y.CompareTo(x));// Slow
        int l = 0, r = nums.Length;
        while(l < r){
            int m = l + (r - l) / 2;
            if(nums[m] <= m) r = m;
            else l = m + 1;
            }
        
        return l < nums.Length && l == nums[l] ? -1 : l;
    }
}
public class Solution {
    public int SpecialArray(int[] nums) {
        nums = nums.OrderBy(c => c).ToArray(); 
        // Array.Sort(nums, (x, y) => y.CompareTo(x));// Slow
        if (nums.Length <= nums[0]) return nums.Length;
       
        for(int i = 1; i<nums.Length;i++){
            if (nums[i-1] < nums.Length-i && nums.Length-i <= nums[i]) 
                return  nums.Length-i;
        }
       
        return -1;
        
    }
}

public class Solution {
    public int SpecialArray(int[] nums) {
         Array.Sort(nums);
        for(int i = 0; i <= nums.Length; i++){
            int l = 0, r = nums.Length;
            while(l < r){
                int m = l + (r - l) / 2;
                if(nums[m] >= i) r = m;
                else l = m + 1;
            }
            if((nums.Length - l) == i) return i;
        }
        return -1;
    }
}
// Day 8
// 1351. Count Negative Numbers in a Sorted Matrix
// Binary Search
public class Solution {
    public int CountNegatives(int[][] grid) {
        int sum = 0;
        foreach(int[] arr in grid){
            sum += BinarySearch(arr);
        }
        return sum;
    }
    
    public int BinarySearch(int[] array) {
        int l = 0, r = array.Length;
        while(l<r){
            int m = l+(r-l)/2;
            if(array[m]<0) r = m;
            else l = m+1;
        }
        return array.Length - r;
    }
}
// Other solution
// Analysis:
// At most move m + n steps.
// Time: O(m + n), space: O(1).
// Start from bottom-left corner of the matrix,
// count in the negative numbers in each row.
public class Solution {
    public int CountNegatives(int[][] grid) {
        int m = grid.Length, n = grid[0].Length, r = m - 1, c = 0, cnt = 0;
        while (r >= 0 && c < n) {
            if (grid[r][c] < 0) {
                --r;
                cnt += n - c; // there are n - c negative numbers in current row.
            }else {
                ++c;
            }
        }
        return cnt;
    }
}

// 74. Search a 2D Matrix
public class Solution {
    public bool SearchMatrix(int[][] matrix, int target) {
        if(matrix.Length ==0) return false;
        int l= 0, r= matrix.Length * matrix[0].Length;
        while(l<r)
        {
            int m = l+(r-l)/2;
            if(matrix[m/matrix[0].Length][m%matrix[0].Length]==target)
            {
                return true;
            }
            else if(matrix[m/matrix[0].Length][m%matrix[0].Length]> target)
            {
                r = m;
            }
            else
            {
                l = m + 1;
            }
        }
        return false;
    }
}    

// Day 9
// 1337. The K Weakest Rows in a Matrix
// Binary Search + PriorityQueue (Max Heap)
public class Solution {
    public int[] KWeakestRows(int[][] mat, int k) {
         PriorityQueue<int[],int[]> pq = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a, b) => a[0] != b[0] ? b[0] - a[0] : b[1] - a[1]));
        int[] ans = new int[k];
        
        for (int i = 0; i < mat.Length; i++) {
            pq.Enqueue(new int[] {numOnes(mat[i]), i},new int[] {numOnes(mat[i]), i});
            if (pq.Count > k)
                pq.Dequeue();
        }
        
        while (k > 0)
            ans[--k] = pq.Dequeue()[1];
        
        return ans;
    }
    
    private int numOnes(int[] row) {
        int l = 0, r = row.Length;
        
        while (l < r) {
            int m = l + (r - l) / 2;
            
            if (row[m] == 1)
                l = m + 1;
            else
                r = m;
        }
        
        return l;
    }
}

//Linq
public class Solution {
    public int[] KWeakestRows(int[][] mat, int k) {
        var list = new List<int[]>();
        for(int i = 0; i < mat.Length; i++)
            list.Add(new int[]{mat[i].Sum(), i});
        list.Sort((x, y) => x[0] == y[0] ? x[1].CompareTo(y[1]) : x[0].CompareTo(y[0]));
        return list.Select(x => x[1]).Take(k).ToArray();
    }
}

// Linq
public class Solution {
    public int[] KWeakestRows(int[][] mat, int k) {
        return mat
		.Select((x, i) => (i, c: x.Count(y => y == 1)))
		.OrderBy(x => x.c)
		.ThenBy(x => x.i)
		.Select(x => x.i)
		.Take(k)
		.ToArray();
    }
}
// 1346. Check If N and Its Double Exist
// Binary Search
public class Solution {
    public bool CheckIfExist(int[] arr) {
        Array.Sort(arr);
        int zeroCount = 0;
        foreach (int x in arr) {
            if (x != 0) {
                if (binarySearch(x, arr) && binarySearch(x*2, arr)) {
                    return true;
                }
            }
            else {
                ++zeroCount;
            }
        }
        return zeroCount >= 2;
    }
    
    public bool binarySearch(int x, int[] nums) {
        int l = 0, r = nums.Length;
        while (l < r) {
            int m = (int)(l+(r-l)/2);
            if (nums[m] < x) {
                l = 1 + m;
            }
            else if (nums[m] > x) {
                r = m;
            }
            else {
                return true;
            }
        }
        return false;
    }
}

// Other Solution
public class Solution {
    public bool CheckIfExist(int[] arr) { 
        List<int> seen = new List<int>();   
        foreach (int i in arr) {
            //i is half or 2 times of a number in seen.
            if (seen.Contains(2 * i) || i % 2 == 0 && seen.Contains(i / 2))
                return true;
            seen.Add(i);
        }
        return false;
    }
}

// Day 10
// 350. Intersection of Two Arrays II
/* Approach : Sort then Two Pointers
Complexity:
Time: O(MlogM + NlogN), where M <= 1000 is length of nums1 array, N <= 1000 is length of nums2 array.
Extra Space (without counting output as space): O(1)
*/
public class Solution {
    public int[] Intersect(int[] nums1, int[] nums2) {
        Array.Sort(nums1);
        Array.Sort(nums2);
        int i = 0, j = 0;
        List<int> list = new List<int>();
        while(i < nums1.Length && j < nums2.Length){
            if(nums1[i] == nums2[j]){
                list.Add(nums1[i]);
                i++;
                j++;
            }else if(nums1[i] < nums2[j]){
                i++;
            }else{
                j++;
            }
        }
        return list.ToArray();
    }
}
// 633. Sum of Square Numbers
// Two Pointer
public class Solution {
    public bool JudgeSquareSum(int c) {
        if(c < 3)
        {
            return true;
        }
        long l = 0 , r = (long)Math.Sqrt(c);
        while(l <= r)
        {
            if((l*l + r*r)==c)
            {
                return true;
            }
            else if((l*l + r*r)<c)
            {
                l++;
            }
            else
            {
                r--;
            }
        }
        return false;
    }
}
// Approach : Binary Search
public class Solution {
    public bool JudgeSquareSum(int c) {
        for (long a = 0; a * a <= c; a++) {
            int b = c - (int)(a * a);
            if (binary_search(0, b, b))
                return true;
        }
        return false;
    }
    
    public bool binary_search(long s, long e, int n) {
        if (s > e)
            return false;
        long m = s + (e - s) / 2;
        if (m * m == n)
            return true;
        if (m * m > n)
            return binary_search(s, m - 1, n);
        return binary_search(m + 1, e, n);
    }
}
// Approach : Fermat Theorem
public class Solution {
    public bool JudgeSquareSum(int c) {
        for (int i = 2; i * i <= c; i++) {
            int count = 0;
            if (c % i == 0) {
                while (c % i == 0) {
                    count++;
                    c /= i;
                }
                if (i % 4 == 3 && count % 2 != 0)
                    return false;
            }
        }
        return c % 4 != 3;
    }
}
// Day 11
// 1855. Maximum Distance Between a Pair of Values
// Binary Search
public class Solution {
    public int MaxDistance(int[] nums1, int[] nums2) {
        int n = nums1.Length;
        int maxi =0;
        for(int i=0;i<n;i++){
            int num = nums1[i];
            int ind = binary_search(nums2,num);
            maxi = Math.Max(maxi,ind-i);
        }
        return maxi;
    }
    int binary_search(int[] nums,int k){
        int l=0,r=nums.Length;
        while(l<r){
            int m = l+(r-l)/2;
            if(nums[m] < k) r = m;
            else l = m+1;
        }
        return l-1;
    
    }
}
/*
Solution 1
Iterate on input array B
Time O(n + m)
Space O(1)
*/
public class Solution {
    public int MaxDistance(int[] nums1, int[] nums2) {
        int res = 0, i = 0, n = nums1.Length, m = nums2.Length;
        for (int j = 0; j < m; ++j) {
            while (i < n && nums1[i] > nums2[j])
                i++;
            if (i == n) break;
            res = Math.Max(res, j - i);
        }
        return res;
    }
}
/*
Solution 2
Iterate on input array A
Time O(n + m)
Space O(1)
*/
public class Solution {
    public int MaxDistance(int[] nums1, int[] nums2) {
        int res = 0, j = -1, n = nums1.Length, m = nums2.Length;
        for (int i = 0; i < n; ++i) {
            while (j + 1 < m && nums1[i] <= nums2[j + 1])
                j++;
            res = Math.Max(res, j - i);
        }
        return res;
    }
}
/*
Solution 2 pointers
Iterate on input array A and B
Time O(n + m)
Space O(1)
*/
public class Solution {
    public int MaxDistance(int[] nums1, int[] nums2) {
         int i = 0, j = 0, res = 0, n = nums1.Length, m = nums2.Length;
        while (i < n && j < m) {
            if (nums1[i] > nums2[j])
                i++;
            else
                res = Math.Max(res, j++ - i);
        }
        return res;
    }
}
// 33. Search in Rotated Sorted Array
// Binary Search
public class Solution {
    public int Search(int[] nums, int target) {
        int l = 0, r = nums.Length;
        while (l < r){
            int m = l+(r-l) / 2;
            if (nums[m] == target)
                return m;
            if ((nums[0] <= target && target < nums[m] )||( target < nums[m] && nums[m] < nums[0]) ||( nums[m] < nums[0]  &&  nums[0]<= target)){
                r = m;
            }
            else{
                l = m+1;
            } 
        }

        return -1;
    }
}
// Day 12
// 153. Find Minimum in Rotated Sorted Array
public class Solution {
    public int FindMin(int[] nums) {
        int l = 0,  r = nums.Length-1;
        
        while(l< r) {
            int m = l+(r-l)/2;
            if(nums[m] > nums[r])
                l = m + 1;
            else if (nums[m] < nums[r])
                r = m;
        }
        return nums[l];
    }
}

// Binary Search II

// Day 1
// 209. Minimum Size Subarray Sum
// Binary Search
// Complexity analysis 
// Time complexity: O(nlog(n)).
// Space complexity: O(n). 
public class Solution {
    public int binarySearch(int[] nums, int l, int r,int target) {
        //find the index of first element that is bigger than or equals target
        while(l<r){
            int m = l+ (r-l)/2;
            if(nums[m] >= target) r = m;
            else l = m +1 ;
        }
        if(r == (nums.Length -1) && nums[r] < target) return -1;
        return r;
    }
    public int MinSubArrayLen(int target, int[] nums) {
    
        int n = nums.Length, len = Int32.MaxValue;
        int[] sums = new int[n + 1]; sums[0] = 0;
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        
        for(int i = 0; i<n; i++){
            int t = target + sums[i];
            int border = binarySearch(sums, 0, n, t);
            if(border>0) len = Math.Min(len,border-i);
        }
        return len == Int32.MaxValue ? 0 : len;
    }
}
// Binary Search
public class Solution {
    public int MinSubArrayLen(int target, int[] nums) {
        int n = nums.Length, len = Int32.MaxValue;
        int[] sums = new int[n + 1]; Array.Fill(sums,0);
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }

        for(int i=n;i>=0;i--){
            int r = i,l = 0;
            int tempAns = Int32.MaxValue;
            while(l <= r){
                int m = l +(r - l)/2;
                if(sums[i] - sums[m] >= target){
                    tempAns = i-m;
                    l = m+1;
                }
                else{
                    r = m-1;
                }
            }
            len=Math.Min(len,tempAns);
        }
        return len==Int32.MaxValue?0:len;
    }
}
/*Sliding Window
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
*/
public class Solution {
    public int MinSubArrayLen(int target, int[] nums) {
        
        int i = 0, n = nums.Length, res = n + 1;
        for (int j = 0; j < n; ++j) {
            target -= nums[j];
            while (target <= 0) {
                res = Math.Min(res, j - i + 1);
                target += nums[i++];
            }
        }
        return res % (n + 1);
    }
}
/*
Solution 1: Two Pointers (Sliding Window)
Maintain a sliding window [l, r) such that sum(nums[l:r)) >= s, 
then move l to l + 1, and move r accordingly to make the window valid.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int MinSubArrayLen(int target, int[] nums) {
        int l = 0;
        int r = 0;
        int t = 0;
        int ans = Int32.MaxValue;
        while (l < nums.Length) {
          while (t < target && r < nums.Length) t += nums[r++];      
          if (t < target) break;
          ans = Math.Min(ans, r - l);      
          t -= nums[l++];
        }
        return ans == Int32.MaxValue ? 0 : ans;  
    }
}
// 611. Valid Triangle Number
//Binary Search
public class Solution {
    public int TriangleNumber(int[] nums) {
         int count = 0;
        Array.Sort(nums);
        for (int i = 0; i < nums.Length - 2; i++) {
            int k = i + 2;
            for (int j = i + 1; j < nums.Length - 1 && nums[i] != 0; j++) {
                k = binarySearch(nums, k, nums.Length - 1, nums[i] + nums[j]);
                count += k - j - 1;
            }
        }
        return count;
    }
    int binarySearch(int[] nums, int l, int r, int x) {
        while (l <= r) {
            int m = l+ (r-l) / 2;
            if (nums[m] >= x)
                r = m - 1;
            else
                l = m + 1;
        }
        return l;
    }
}

/* 3 pointers
Same as https://leetcode.com/problems/3sum-closest
Assume a is the longest edge, b and c are shorter ones, to form a triangle, they need to satisfy len(b) + len(c) > len(a).
*/ 
public class Solution {
    public int TriangleNumber(int[] nums) {
        int result = 0;
        if (nums.Length < 3) return result;
        
        Array.Sort(nums);

        for (int i = 2; i < nums.Length; i++) {
            int l = 0, r = i - 1;
            while (l< r) {
                // for each pair where you have a[l] + a[r]
                // is greater than a[i], you know that for all
                // indices greater l up until r - would satisfy
                // the triangle condition = and you subtract r
                // by 1 to look for the next
                // else increase l by 1 and add nothing to rslt.
                if (nums[l] + nums[r] > nums[i]) {
                    result += (r - l);
                    r--;
                }
                else {
                    l++;
                }
            }
        }
        
        return result;
    }
}
// Two-Pointer
/*
Complexity
Time: O(N^2), where N <= 1000 is number of elements in the array nums.
Space: O(logN), logN is the space complexity for sorting.
*/
public class Solution {
    public int TriangleNumber(int[] nums) {
        Array.Sort(nums);
        int n = nums.Length, ans = 0;
        for (int k = 2; k < n; ++k) {
            int i = 0, j = k - 1;
            while (i < j) {
                if (nums[i] + nums[j] > nums[k]) {
                    ans += j - i;
                    j -= 1;
                } else {
                    i += 1;
                }
            }
        }
        return ans;
    }
}
/*
Idea: Greedy
Time Complexity: O(n^2)
*/
public class Solution {
    public int TriangleNumber(int[] nums) {
         if (nums.Length < 3) return 0;
        nums = nums.OrderByDescending(c => c).ToArray();
        
        int n = nums.Length;
        int ans = 0;
        for (int c = 0; c < n-2; ++c) {        
            int b = c + 1;
            int a = n - 1;
            while (b < a) {
                if (nums[a] + nums[b] > nums[c]) {
                    ans += (a - b);
                    ++b;
                } else {
                    --a;
                }
            }
        }
        
        return ans;
    }
}
// Day 2
// 658. Find K Closest Elements
/* Sliding Window
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
*/
public class Solution {
    public IList<int> FindClosestElements(int[] arr, int k, int x) {
        int l = 0, r = arr.Length - k;
        while (l < r) {
            int m = l+ (r - l) / 2;
            if (x - arr[m] > arr[m + k] - x)
                l = m + 1;
            else
                r = m;
        }
        return arr[l..(l + k)].ToList();
    }
}

// Two-Pointer
public class Solution {
    public IList<int> FindClosestElements(int[] arr, int k, int x) {
        int l = 0, r = arr.Length - 1;
        while (r - l >= k) {
            if (arr[r] - x >= x - arr[l]) {
                r--;
            } else {
                l++;
            }
        }
        List<int> res = new List<int>();
        for (int i = l; i <= r; i++) res.Add(arr[i]);
        return res;
    }
}
// 1894. Find the Student that Will Replace the Chalk
/*
Time O(n)
Space O(1)
*/
public class Solution {
    public int ChalkReplacer(int[] chalk, int k) {
        long sum=0;
        for(int i=0;i<chalk.Length;i++) sum+=chalk[i];
        int left=(int)(k%sum);
        for(int i=0;i<chalk.Length;i++) {
            if(left<chalk[i]) return i;
            left-=chalk[i];
        }
        return 0;
    }
}
//  Two passes O(n) 
// 1st pass used to avoid possible int overflow.
public class Solution {
    public int ChalkReplacer(int[] chalk, int k) {
        int sum = 0;
        for (int i = 0; i < chalk.Length; ++i) {
            sum += chalk[i];
            k -= chalk[i];
            if (k < 0) {
                return i;
            }
        }
        k %= sum; 
        for (int i = 0; i < chalk.Length; ++i) {
            k -= chalk[i];
            if (k < 0) {
                return i;
            }
        }
        return 0;
    }
}
// Day 3
// 300. Longest Increasing Subsequence
/*
Solution 2: DP + Binary Search / Patience Sort

dp[i] := smallest tailing number of a increasing subsequence of length i + 1.
dp is an increasing array, we can use binary search to find the index to insert/update the array.
ans = len(dp)

Time complexity: O(nlogn)
Space complexity: O(n)
*/
public class Solution {
    public int LengthOfLIS(int[] nums) {       
    int n = nums.Length;
    if (n == 0) return 0;    
    List<int> dp = new List<int>();
    for (int i = 0; i < n; ++i) {
      int it = binarySearch(dp, nums[i]);
        if (it < 0) it = ~it;
      if (it == dp.Count)
        dp.Add(nums[i]);
      else
        dp[it] = nums[i];
    }
    return dp.Count;
    
    }
    
    int binarySearch(List<int> nums, int x) {
        int l = 0, r = nums.Count;
        while (l < r) {
            int m = l+ (r-l) / 2;
            if (nums[m] >= x)
                r = m;
            else
                l = m + 1;
        }
        return l;
    }
}

// Use C# Binary Seach + Patience Sorting
// This algorithm is actually Patience sorting.
// It might be easier for you to understand how it works if you think about it as piles of cards instead of tails.
// The number of piles is the length of the longest subsequence.
// For more info see Princeton lecture.
public class Solution {
    public int LengthOfLIS(int[] nums) {
        List<int> dp = new List<int>();
        
        foreach (int x in nums) {
            int it = dp.BinarySearch(x); // directly use C# binary search here
            if (it < 0) { it = ~it;}
            if (it == dp.Count){
                dp.Add(x);   
            }
            else{
                dp[it] = x;
            }
        }
        return dp.Count;
    }
}
// Binary Search / Patience sorting
public class Solution {
    public int LengthOfLIS(int[] nums) {
        int[] tails = new int[nums.Length];
        int size = 0;
        foreach (int x in nums) {
            int i = 0, j = size;
            while (i != j) { // use binary search here!
                int m = (i + j) / 2;
                if (tails[m] < x)
                    i = m + 1;
                else
                    j = m;
            }
            tails[i] = x;
            if (i == size) ++size;
        }
        return size;
    }
}
/*
Dynamic Programming
Time Complexity: O(n^2)
sort of slow
*/
public class Solution {
    public int LengthOfLIS(int[] nums) {
        if (nums.Length == 0) return 0;
        int n = nums.Length;
        int[] f = new int[n]; Array.Fill(f, 1);
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j)
                if (nums[i] > nums[j])
                    f[i] = Math.Max(f[i], f[j] + 1);
        return f.Max();
       
    }
}

// 1760. Minimum Limit of Balls in a Bag
/*Binary Search
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
Space O(1)*/
public class Solution {
    public int MinimumSize(int[] nums, int maxOperations) {
        int l = 1, r = 1_000_000_000;
        while (l < r) {
            int m = l + (r - l) / 2, count = 0;
            foreach (int a in nums)
                count += (a - 1) / m;
            if (count > maxOperations)
                l = m + 1;
            else
                r = m;
        }
        return l;
    }
}
/*
Find the smallest penalty that requires less or equal ops than max_ops.

Time complexity: O(nlogm)
Space complexity: O(1)
*/
public class Solution {
    public int MinimumSize(int[] nums, int maxOperations) {
       int l = 1, r = nums.Max();
    while (l < r) {
      int m = l + (r - l) / 2;
      int count = 0;
      foreach (int x in nums) 
        count += (x - 1) / m;
      if (count <= maxOperations)
        r = m;
      else
        l = m + 1;
    }
    return l;
    }
}

// Day 4
// 875. Koko Eating Bananas
/*
Solution: Binary Search
search for the smallest k [1, max_pile_height] such that eating time h <= H.

Time complexity: O(nlogh)
Space complexity: O(1)
*/
public class Solution {
    public int MinEatingSpeed(int[] piles, int h) {
        int l = 1;
    int r = piles.Max() + 1;
    while (l < r) {
      int m = (r - l) / 2 + l;
      int a = 0;
      foreach (int p in piles)
        a += (p + m - 1) / m;
      if (a <= h)
        r = m;
      else
        l = m + 1;
    }
    return l;
    }
}

// 1552. Magnetic Force Between Two Balls
/*
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
*/
public class Solution {
    public int MaxDistance(int[] position, int m) {
        Array.Sort(position);
        int l = 0, r = position[position.Length-1] - position[0];
        while (l <= r) {
            int mid = l +  (r - l) / 2;
            if (count(position, mid) >= m)
                l = mid + 1;
            else
                r = mid - 1;
        }
         //You can try out the dry run for example position = [1,2,3,4,7], m = 3 ,ans will be one position less than left
        return l-1;

    }
private int count(int[] position, int d) {
        int ans = 1, cur = position[0];
        for (int i = 1; i < position.Length; ++i) {
            if (position[i] - cur >= d) {
                ans++;
                cur = position[i];
            }
        }
        return ans;
    }
}
/*Solution: Binary Search
Find the max distance that we can put m balls.

Time complexity: O(n*log(distance))
Space complexity: O(1)*/
public class Solution {
    public int MaxDistance(int[] position, int m) {
        Array.Sort(position);
    int n = position.Length;
    int l = 0;
    int r = position[n - 1] - position[0] + 1;
    int t = r;
    while (l < r) {
      int mid = l + (r - l) / 2;
      if (getCount(position, t - mid) >= m)
        r = mid;
      else
        l = mid + 1;
    }
    return t - l;
  }
  
  private int getCount(int[] position, int d) {
    int count = 1;
    int last = position[0];
    foreach (int x in position) {
      if (x - last >= d) {
        ++count;
        last = x;
      }
    }
    return count;
    }
}
// Day 5
// 287. Find the Duplicate Number
/*Explanation
Binary search the result.
If the sum > threshold, the divisor is too small.
If the sum <= threshold, the divisor is big enough.

Complexity
Time O(NlogM), where M = max(A)
Space O(1)*/
public class Solution {
    public int SmallestDivisor(int[] nums, int threshold) {
        int l = 1, r = (int)1e6;
        while (l < r) {
            int m = l + (r - l) / 2, sum = 0;
            foreach (int i in nums)
                sum += (i + m - 1) / m;
            if (sum > threshold)
                l = m + 1;
            else
                r = m;
        }
        return l;
    }
}
/*
Solution : Binary Search
Time complexity: O(nlogn)
Space complexity: O(1)

Find the smallest m such that len(nums <= m) > m, which means m is the duplicate number.
In the sorted form [1, 2, …, m1, m2, m + 1, …, n]
There are m+1 numbers <= m
*/
public class Solution {
    public int FindDuplicate(int[] nums) {
       int l = 1;
    int r = nums.Length;
    while (l < r) {
      int m = (r - l) / 2 + l;
      int count = 0; // len(nums <= m)
      foreach (int num in nums)
        if (num <= m) ++count;
      if (count <= m)
        l = m + 1;
      else
        r = m;
    }
    return l;
    }
}
/*
Solution: Linked list cycle
Convert the problem to find the entry point of the cycle in a linked list.
Take the number in the array as the index of next node.

[1,3,4,2,2]
0->1->3->2->4->2 cycle: 2->4->2
[3,1,3,4,2]
0->3->4->2->3->4->2 cycle 3->4->2->3

Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public int FindDuplicate(int[] nums) {
       int slow = 0;
    int fast = 0;
    while (true) {
      slow = nums[slow];
      fast = nums[nums[fast]];
      if (slow == fast) break;
    }
    fast = 0;
    while (fast != slow) {
      slow = nums[slow];
      fast = nums[fast];
    }
    return slow;
    }
}
// 1283. Find the Smallest Divisor Given a Threshold
/* Approach : Binary Search

Complexity Analysis
Time Complexity: O(nlogn)
The outer loop uses binary search to identify a candidate - this runs in O(\log n)O(logn) time. 
For each candidate, we iterate over the entire array which takes O(n)O(n) time, 
resulting in a total of O(n \log n)O(nlogn) time.
Space Complexity: O(1) */
public class Solution {
    public int FindDuplicate(int[] nums) {
        // 'low' and 'high' represent the range of values of the target        
        int l = 1, r = nums.Length ;
        int duplicate = -1;
        
        while (l < r) {
            int cur = l + (r - l) / 2;

            // Count how many numbers in 'nums' are less than or equal to 'cur'
            int count = 0;
            foreach (int num in nums) {
                if (num <= cur)
                    count++;
            }
            
            if (count > cur) {
                duplicate = cur;
                r = cur;
            } else {
                l = cur + 1;
            }
        }
        return duplicate;
    }
}
/*
Solution: Binary Search

Time complexity: O(nlogk)
Space complexity: O(1)
*/
public class Solution {
    public int SmallestDivisor(int[] nums, int threshold) {
       int sums(int d) {
      int s = 0;
      foreach (int n in nums)
        s += (n + (d - 1)) / d;
      return s;
    };
    int l = 1;
    int r = 1000000;
    while (l < r) {
      int m = l + (r - l) / 2;
      if (sums(m) <= threshold)
        r = m;
      else
        l = m + 1;
    }
    return l;
  
    }
}

// Day 6
// 1898. Maximum Number of Removable Characters
public class Solution {
    public int MaximumRemovals(string s, string p, int[] removable) {
        int l = 0, r = removable.Length;
        int[] map = new int[s.Length];
        Array.Fill(map, removable.Length);
        for (int i = 0; i < removable.Length; ++i)
             map[removable[i]] = i;    
        while (l < r) {
            int m = (l + r + 1) / 2, j = 0;
            for (int i = 0; i < s.Length && j < p.Length; ++i)
                if (map[i] >= m && s[i] == p[j])
                    ++j;
            if (j == p.Length)
                l = m;
            else
                r = m - 1;
        }
        return l;
    }
}
/*
Solution: Binary Search + Two Pointers
If we don’t remove any thing, p is a subseq of s, as we keep removing, 
at some point L, p is no longer a subseq of s. 
e.g [0:True, 1: True, …, L – 1: True, L: False, L+1: False, …, m:False], this array is monotonic. 
We can use binary search to find the smallest L such that p is no long a subseq of s. Ans = L – 1.

For each guess, we can use two pointers to check whether p is subseq of removed(s) in O(n).

Time complexity: O(nlogn)
Space complexity: O(n)
*/
public class Solution {
    public int MaximumRemovals(string s, string p, int[] removable) {
    int n = s.Length;
    int t = p.Length;
    int m = removable.Length;    
    int l = 0;
    int r = m + 1;
        int[] idx = new int[n]; Array.Fill(idx,Int32.MaxValue);
    for (int i = 0; i < m; ++i)
      idx[removable[i]] = i;
    while (l < r) {
      int mid = l + (r - l) / 2;
      int j = 0;
      for (int i = 0; i < n && j < t; ++i)
        if (idx[i] >= mid && s[i] == p[j]) ++j;      
      if (j != t)
        r = mid;
      else
        l = mid + 1;
    }
    // l is the smallest number s.t. p is no longer a subseq of s.
    return l - 1;
    }
}
// 1870. Minimum Speed to Arrive on Time
public class Solution {
    public int MinSpeedOnTime(int[] dist, double hour) {
        if (hour <= dist.Length - 1)
            return -1;
        int l = 1, r = 10000000;
        while (l < r) {
            int m = l + (r - l) / 2, time = 0;
            for (int i = 0; i < dist.Length - 1; ++i)
                time += dist[i] / m + (dist[i] % m > 0 ? 1 : 0);
            if ((double)dist[dist.Length - 1] / m + time <= hour)
                r = m;
            else
                l = m + 1;
        }
        return l;
        
    }
}
/*
Solution: Binary Search
l = speedmin=1
r = speedmax+1 = 1e7 + 1

Find the min valid speed m such that t(m) <= hour.

Time complexity: O(nlogn)
Space complexity: O(1)
*/
public class Solution {
    public int MinSpeedOnTime(int[] dist, double hour) {
        int kMax = 10000000 + 1;
        int n = dist.Length;    
        int l = 1;
        int r = kMax;    
        while (l < r) {
          int m = l + (r - l) / 2;
          int t = 0;
          for (int i = 0; i < n - 1; ++i)
            t += (dist[i] + m - 1) / m;
          if (t + dist[dist.Length - 1] * 1.0 / m <= hour)
            r = m;
          else
            l = m + 1;      
        }
        return l == kMax ? -1 : l;
        
    }
}
// Day 7
// 1482. Minimum Number of Days to Make m Bouquets
/*Binary Search 
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

Though I don't thik worth doing that.*/
public class Solution {
    public int MinDays(int[] bloomDay, int m, int k) {
        int n = bloomDay.Length, l = 1, r = (int)1e9;
        if (m * k > n) return -1;
        while (l < r) {
            int mid = l + (r - l) / 2, flow = 0, bouq = 0;
            for (int j = 0; j < n; ++j) {
                if (bloomDay[j] > mid) {
                    flow = 0;
                } else if (++flow >= k) {
                    bouq++;
                    flow = 0;
                }
            }
            if (bouq < m) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
}
/*
Solution: Binary Search
Find the smallest day D that we can make at least m bouquets using binary search.

at a given day, we can check how many bouquets we can make in O(n)

Time complexity: O(nlog(max(days))
Space complexity: O(1)
*/
public class Solution {
    public int MinDays(int[] bloomDay, int m, int k) {
    int kInf = bloomDay.Max() + 1;
    int l = bloomDay.Min();
    int r = kInf;
    
    // Return the number of bouquets we can get at day D.
    int getBouquets(int D) {
      int ans = 0;
      int cur = 0;
      foreach (int d in bloomDay) {
        if (d > D) {
          cur = 0;
        } else if (++cur == k) {
          ++ans;
          cur = 0;          
        }
      }
      return ans;
    };
          
    while (l < r) {
      int mid = l + (r - l) / 2;      
      // Find smallest day that bouquets >= m.
      if (getBouquets(mid) >= m)
        r = mid;
      else
        l = mid + 1;
    }
    return l >= kInf ? -1 : l;
    }
}

// 1818. Minimum Absolute Sum Difference
/*
Clone nums1 and sort it as sorted1;
Traverse nums2, for each number find its corresponding difference from nums1: diff = abs(nums1[i] - nums2[i]) and use binary search against sorted1 to locate the index, idx, at which the number has minimum difference from nums2[i], then compute minDiff = abs(sorted1[idx] - nums2[i]) or abs(sorted1[idx - 1] - nums2[i]); Find the max difference mx out of all diff - minDiff;
Use the sum of the corresponding difference to substract mx is the solution.

Analysis:
Time: O(nlogn), space: O(n), where n = nums1.length = nums2.length.
*/
public class Solution {
    public int MinAbsoluteSumDiff(int[] nums1, int[] nums2) {
     long sum = 0,ans;
		int mod = 1000000007;
		int[] nums = new int[nums1.Length];

		for(int i = 0; i < nums.Length; i++)
		{
			sum = (sum + Math.Abs(nums1[i] - nums2[i]));
			nums[i] = nums1[i];
		}

		ans = sum;
		Array.Sort(nums1);

		for(int i = 0; i < nums.Length; i++)
		{
			int idx = Array.BinarySearch(nums1, nums2[i]);
			if (idx < 0)
				idx = ~idx;

			int left = idx > 0 ? Math.Abs(nums1[idx - 1] - nums2[i]) : int.MaxValue;
			int right = idx < nums.Length ? Math.Abs(nums1[idx] - nums2[i]) : int.MaxValue;
			long temp = (sum - Math.Abs(nums[i] - nums2[i]) + Math.Min(left, right));

			ans = Math.Min(ans, temp);
		}

		return (int)(ans % mod);
       
    }
}
// O(NLogN) time compexicity:
// In this solution we impelement binary search ourselves
public class Solution {
    public int MinAbsoluteSumDiff(int[] nums1, int[] nums2) {
        var sorted = nums1.Distinct().OrderBy(n => n).ToArray();
        var maxDiff = -1;
        var maxIndex = -1;
        var maxReplacement = -1;
        for (int i = 0; i < nums1.Length; i++)
        {
            var originalDiff = Math.Abs(nums1[i] - nums2[i]);
            if (originalDiff == 0)
                continue;
            
            var replacement = GetClosestNumber(sorted, nums2[i]);
            var diff = Math.Abs(originalDiff - Math.Abs(replacement - nums2[i]));
            if (diff > maxDiff)
            {
                maxDiff = diff;
                maxIndex = i;
                maxReplacement = replacement;
            }
        }
        
        var result = 0;
        for (int i = 0; i < nums1.Length; i++)
        {
            var curr = i == maxIndex ? maxReplacement : nums1[i];
            result = (result + Math.Abs(curr - nums2[i])) % (int)(1e9 + 7);
        }
        return result;
    }
    
    private int GetClosestNumber(int[] sorted, int target)
    {
        var left = 0;
        var right = sorted.Length - 1;
        while (left < right)
        {
            var mid = left + (right - left) / 2;
            var midDiff = Math.Abs(sorted[mid] - target);
            var nextDiff = Math.Abs(sorted[mid + 1] - target);
            
            if (midDiff > nextDiff)
                left = mid + 1;
            else if (midDiff < nextDiff)
                right = mid;
            else
                return sorted[mid];
        }
        return sorted[left];
       
    }
}
// O(NLogN) time compexicity:
public class Solution {
    public int MinAbsoluteSumDiff(int[] nums1, int[] nums2) {
     var sorted = nums1.Distinct().OrderBy(n => n).ToList();
        var maxDiff = -1;
        var maxIndex = -1;
        var maxReplacement = -1;
        for (int i = 0; i < nums1.Length; i++)
        {
            var originalDiff = Math.Abs(nums1[i] - nums2[i]);
            if (originalDiff == 0)
                continue;
            
            var replacement = GetReplacement(sorted, sorted.BinarySearch(nums2[i]), nums2[i]);
            var diff = Math.Abs(originalDiff - Math.Abs(replacement - nums2[i]));
            if (diff > maxDiff)
            {
                maxDiff = diff;
                maxIndex = i;
                maxReplacement = replacement;
            }
        }
        
        var result = 0;
        for (int i = 0; i < nums1.Length; i++)
        {
            var curr = i == maxIndex ? maxReplacement : nums1[i];
            result = (result + Math.Abs(curr - nums2[i])) % (int)(1e9 + 7);
        }
        return result;
    }
    
    private int GetReplacement(List<int> nums, int index, int target)
    {
        if (index >= 0)
            return target;
        
        index = ~index;
        if (index == 0)
            return nums[index];
        
        if (index == nums.Count)
            return nums[index - 1];
        
        return Math.Abs(nums[index] - target) < Math.Abs(nums[index - 1] - target) ? nums[index] : nums[index - 1];
    
       
    }
}
// Day 8
// 240. Search a 2D Matrix II
/*We start search the matrix from top right corner, 
initialize the current position to top right corner, 
if the target is greater than the value in current position, 
then the target can not be in entire row of current position because the row is sorted, 
if the target is less than the value in current position, 
then the target can not in the entire column because the column is sorted too. 
We can rule out one row or one column each time, so the time complexity is O(m+n).

Solution : Two Pointers
Start from first row + last column, if the current value is larger than target, –column; if smaller then ++row.

e.g.
1. r = 0, c = 4, v = 15, 15 > 5 => –c
2. r = 0, c = 3, v = 11, 11 > 5 => –c
3. r = 0, c = 2, v = 7, 7 > 5 => –c
4. r = 0, c = 1, v = 4, 4 < 5 => ++r
5. r = 1, c = 1, v = 5, 5 = 5, found it!

Time complexity: O(m + n)
Space complexity: O(1)
*/
public class Solution {
    public bool SearchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.Length < 1 || matrix[0].Length < 1)
        return false;

        int col = 0;
        int row = matrix.Length - 1;
        while (col <= matrix[0].Length - 1 && row >= 0) {
            if (target == matrix[row][col])
                return true;
            else if (target < matrix[row][col])
                row--;
            else if (target > matrix[row][col])
                col++;
        }
        return false;
    }
}
// For those who may wonder, it's also OK to search from the bottom-left point. Just check the code below.
// Actually it's like the matrix contains two "binary search tree" and it has two "roots" correspondingly. :-)
public class Solution {
    public bool SearchMatrix(int[][] matrix, int target) {
        if(matrix == null || matrix.Length < 1 || matrix[0].Length <1) {
            return false;
        }
        int col = matrix[0].Length-1;
        int row = 0;
        while(col >= 0 && row <= matrix.Length-1) {
            if(target == matrix[row][col]) {
                return true;
            } else if(target < matrix[row][col]) {
                col--;
            } else if(target > matrix[row][col]) {
                row++;
            }
        }
        return false;
    }
}
// 275. H-Index II
/*Just binary search, each time check citations[mid]
case 1: citations[mid] == len-mid, then it means there are citations[mid] papers that have at least citations[mid] citations.
case 2: citations[mid] > len-mid, then it means there are citations[mid] papers that have moret than citations[mid] citations, so we should continue searching in the left half
case 3: citations[mid] < len-mid, we should continue searching in the right side
After iteration, it is guaranteed that right+1 is the one we need to find (i.e. len-(right+1) papars have at least len-(righ+1) citations)

Solution : Binary Search
Time Complexity: O(logn)
Space Complexity: O(1)
*/
public class Solution {
    public int HIndex(int[] citations) {
       int l = 0, len = citations.Length, r = len, m;
        while(l < r )
        {
            m = l + (r - l)/2;
            if(citations[m] >= (len-m)) r = m;
            else l = m + 1;
        }
        return len - l;
    }
}
// Day 9
// 1838. Frequency of the Most Frequent Element
/* Sliding Window
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


Solution : Use while loop
*/
public class Solution {
    public int MaxFrequency(int[] nums, int k) {
        int res = 1, i = 0, j;
        long sum = 0;
        Array.Sort(nums);
        for (j = 0; j < nums.Length; ++j) {
            sum += nums[j];
            while (sum + k < (long)nums[j] * (j - i + 1)) {
                sum -= nums[i];
                i += 1;
            }
            res = Math.Max(res, j - i + 1);
        }
        return res;
    }
}
/*
Solution : Use if clause
Just save some lines and improve a little time.
*/
public class Solution {
    public int MaxFrequency(int[] nums, long k) {
        int i = 0, j;
        Array.Sort(nums);
        for (j = 0; j < nums.Length; ++j) {
            k += nums[j];
            if (k < (long)nums[j] * (j - i + 1))
                k -= nums[i++];
        }
        return j - i;
    }
}
/*
Sort the elements, maintain a window such that it takes at most k ops to make the all the elements equal to nums[i].

Time complexity: O(nlogn)
Space complexity: O(1)
*/
public class Solution {
    public int MaxFrequency(int[] nums, long k) {
       Array.Sort(nums);
    int l = 0;
    long sum = 0;
    int ans = 0;
    for (int r = 0; r < nums.Length; ++r) {
      sum += nums[r];      
      while (l < r && 
             sum + k < (long)(nums[r]) * (r - l + 1))
        sum -= nums[l++];
      ans = Math.Max(ans, r - l + 1);
    }
    return ans;
    }
}
// 540. Single Element in a Sorted Array
public class Solution {
    public int SingleNonDuplicate(int[] nums) {
        int l = 0, r = nums.Length-1;
        while(l < r){
            int m = l + (r - l)/2;
            if( (m % 2 == 0 && nums[m] == nums[m +1]) || (m %2 == 1 && nums[m] == nums[m - 1]) )
                l = m + 1;
            else
                r = m;
        }
        return nums[l];
    }
}
// Day 10
// 222. Count Complete Tree Nodes
/*
I denoted values for nodes in the way of how we are going to count them, note that it does not matter in fact what is inside.
First step is to find the number of levels in our tree, you can see, that levels with depth 0,1,2 are full levels and level with depth = 3 is not full here.
So, when we found that depth = 3, we know, that there can be between 8 and 15 nodes when we fill the last layer.
How we can find the number of elements in last layer? We use binary search, because we know, that elements go from left to right in complete binary tree. 
To reach the last layer we use binary decoding,
 for example for number 10, we write it as 1010 in binary, remove first element (it always will be 1 and we not interested in it), and now we need to take 3 steps: 
 010, which means left, right, left.
Complexity. To find number of layers we need O(log n). We also need O(log n) iterations for binary search, on each of them we reach the bottom layer in O(log n).
 So, overall time complexity is O(log n * log n). Space complexity is O(log n).

Code I use auxiliary funcion Path, which returns True if it found node with given number and False in opposite case.
 In main function we first evaluate depth, and then start binary search with interval 2^depth, 2^{depth+1} - 1. We also need to process one border case, where last layer is full.
*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int CountNodes(TreeNode root) {
        if (root == null) {
        return 0;
    }
    int l = leftHeight(root.left);
    int r = leftHeight(root.right);
    if (l == r) { // left side is full
        return CountNodes(root.right) + (1<<l);
    } 
    return CountNodes(root.left) + (1<<r);
}

private int leftHeight(TreeNode node) {
    int h = 0;
    while (node != null) {
        h++;
        node = node.left;
    }
    return h;
    }
}
/*
For each node, count the height of it’s left and right subtree by going left only.

Let L = height(left) R = height(root), if L == R, which means the left subtree is perfect.
It has (2^L – 1) nodes, +1 root, we only need to count nodes of right subtree recursively.
If L != R, L must be R + 1 since the tree is complete, which means the right subtree is perfect.
It has (2^(L-1) – 1) nodes, +1 root, we only need to count nodes of left subtree recursively.

Time complexity: T(n) = T(n/2) + O(logn) = O(logn*logn)

Space complexity: O(logn)
*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int CountNodes(TreeNode root) {
        if (root == null) return 0;
    int l = depth(root.left);
    int r = depth(root.right);
    if (l == r) 
      return (1 << l) + CountNodes(root.right);
    else
      return (1 << (l - 1)) + CountNodes(root.left);
  }
private int depth(TreeNode root) {
    if (root == null) return 0;
    return 1 + depth(root.left);
  
    }
}
// 1712. Ways to Split Array Into Three Subarrays
/*
The approach is not too hard, but the implementation was tricky for me to get right.

First, we prepare the prefix sum array, so that we can compute subarray sums in O(1). Then, we move the boundary of the first subarray left to right. This is the first pointer - i.

For each point i, we find the minimum (j) and maximum (k) boundaries of the second subarray:

nums[j] >= 2 * nums[i]
nums[sz - 1] - nums[k] >= nums[k] - nums[i]
Note that in the code and examples below, k points to the element after the second array. In other words, it marks the start of the (shortest) third subarray. This makes the logic a bit simpler.

With these conditions, sum(0, i) <= sum(i + 1, j), and sum(i + 1, k - 1) < sum(k, n). Therefore, for a point i, we can build k - j subarrays satisfying the problem requirements.

Final thing is to realize that j and k will only move forward, which result in a linear-time solution.

The following picture demonstrate this approach for the [4,2,3,0,3,5,3,12] test case.
*/
public class Solution {
    public int WaysToSplit(int[] nums) {
        int sz = nums.Length, res = 0;
        for (int i = 1; i < sz; ++i)
            nums[i] += nums[i - 1];
        for (int i = 0, j = 0, k = 0; i < sz - 2; ++i) {
            while (j <= i || (j < sz - 1 && nums[j] < nums[i] * 2))
                ++j;
            while (k < j || ( k < sz - 1 && nums[k] - nums[i] <= nums[sz - 1] - nums[k]))
                ++k;
            res = (res + k - j) % 1000000007;
        }    
        return res;
    }
}
// Day 11
// 826. Most Profit Assigning Work
/*Solution 
zip difficulty and profit as jobs.
sort jobs and sort 'worker'.
Use 2 pointers. For each worker, find his maximum profit best he can make under his ability.

Because we have sorted jobs and worker,
we will go through two lists only once.
this will be only O(D + W).
O(DlogD + WlogW), as we sort jobs.
*/
public class Solution {
    public int MaxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
        List<KeyValuePair<int, int>> jobs = new List<KeyValuePair<int, int>>();
        int N = profit.Length, res = 0, i = 0, best = 0;
        for (int j = 0; j < N; ++j)
            jobs.Add(new KeyValuePair<int, int>(difficulty[j], profit[j]));
        jobs.Sort((x, y) => (x.Key.CompareTo(y.Key)));
        Array.Sort(worker);
        foreach (int ability in worker) {
            while (i < N && ability >= jobs[i].Key)
                best = Math.Max(jobs[i++].Value, best);
            res += best;
        }
        return res;
    }
}
/*
Solution : Bucket + Greedy
Key idea: for each difficulty D, find the most profit job whose requirement is <= D.

Three steps:

for each difficulty D, find the most profit job whose requirement is == D, best[D] = max{profit of difficulty D}.
if difficulty D – 1 can make more profit than difficulty D, best[D] = max(best[D], best[D – 1]).
The max profit each worker at skill level D can make is best[D].
Time complexity: O(n)

Space complexity: O(10000)
*/
public class Solution {
    public int MaxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
          const int N = 100000;
        // max profit at difficulty i
        int[] max_profit = new int[N+1]; Array.Fill(max_profit, 0);
        for (int i = 0; i < difficulty.Length; ++i)
          max_profit[difficulty[i]] = Math.Max(max_profit[difficulty[i]], profit[i]);
        for (int i = 2; i <= N; ++i)
          max_profit[i] = Math.Max(max_profit[i], max_profit[i - 1]);
        int ans = 0;
        foreach (int level in worker)
          ans += max_profit[level];
        return ans;
	}
}

/*
Solution 2
Use a dictionary<difficulty, profit>
Go through the treemap once, find the max profit best for each difficulty.
Time O(DlogD + WlogD)
*/
public class Solution {
    public int MaxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
         //this dictionary store {difficulty, maxProfit} pairs
            var dict=new Dictionary<int, int>();
            for(int i = 0; i < difficulty.Length; i++)
            {
                if (!dict.ContainsKey(difficulty[i])) dict.Add(difficulty[i],0);
                dict[difficulty[i]] = Math.Max(dict[difficulty[i]], profit[i]);//get the max profit of this difficulty
            }

            var keys = dict.Keys.OrderBy(x=>x).ToList();//sort asc
            int[] maxProfit = new int[keys.Count];//create maxProfit array
            int max = 0;
            for(int i = 0; i < keys.Count; i++)
            {
                max = Math.Max(max, dict[keys[i]]);
                maxProfit[i] = max;// max of all profit <= current difficulty
            }

            int res = 0;
            foreach(var w in worker)
            {
                if (w < keys[0]) continue;
                if (w >= keys.Last())
                {
                    res += maxProfit.Last();
                }
                else
                {
                    int left = 0;
                    int right = keys.Count - 1;
					//using binary search , get the maxProfix this worker can get, aka w>=keys[i], i is the index
                    while (left < right)
                    {
                        var mid = (left + right+1) / 2;
                        if (w>=keys[mid])
                        {
                            left = mid;
                        }
                        else
                        {
                            right = mid-1;
                        }
                    }
                    res += maxProfit[left];
                }
            }
            return res;
	}
}
//Binary Search
public class Solution
{
	public int MaxProfitAssignment(int[] difficulty, int[] profit, int[] worker)
	{
		int ans = 0;
		List<Node> list = new List<Node>();

		for (int i = 0; i < difficulty.Length; i++)
			list.Add(new Node(difficulty[i], profit[i]));

		list.Sort((x, y) => { 
			return x.difficulty.CompareTo(y.difficulty);
		});

		for(int i = 1; i < list.Count; i++) {
			if(list[i].profit < list[i - 1].profit)
				list[i].profit = list[i - 1].profit;
		}

		foreach (var item in worker)
			ans += BinarySearch(list, item);

		return ans;
	}

	public int BinarySearch(List<Node> list, int num)
	{
		int low = 0, high = list.Count - 1, ans = 0;

		while(low <= high)
		{
			int mid = low + (high - low) / 2;
			if (list[mid].difficulty <= num)
			{
				ans = Math.Max(ans, list[mid].profit);
				low = mid + 1;
			}
			else
				high = mid - 1;
		}

		return ans;
	}
}

public class Node {
	public int difficulty;
	public int profit;

	public Node(int difficulty, int profit) {
		this.difficulty = difficulty;
		this.profit = profit;
	}
}

// 436. Find Right Interval
/*O(n logn) solution - HashMap + Sort + Binary Search
Create an array of starting points where sp[i] = starting point of interval i.
Use a HashMap to keep track of index of each starting point (Each starting point is unique).
Sort the array of starting points
Then for each end point 'ep':
perform binary search on array of starting points to get the minimum start point 'sp' such that sp >= ep
if such a start point is found, add its index (use the HashMap to get its index) to result array, else add -1 to result array
*/
public class Solution {
    public int[] FindRightInterval(int[][] intervals) {
          
        Dictionary<int, int> map = new Dictionary<int,int>();
        int m = intervals.Length;
        int n = intervals[0].Length;
        int[] sp = new int[m];                  //array of starting points
        
        for(int i = 0; i < m; i++) {
            sp[i] = intervals[i][0];            
            map[sp[i]]= i;                  //(key=start_point, val=index)
        }
        
        Array.Sort(sp);                        //sort array of starting points
        int[] result = new int[m];
        
        for(int i = 0; i < m; i++) {
            int l = 0, r = m - 1;
            bool found = false;              //to see if result was found
            int min = -1;
            int ep = intervals[i][n - 1];       //ep = endpoint
            while(l <= r) {                     //binarySearch on arr of start points
                int mid = l + (r - l) / 2;
                if(sp[mid] >= ep) {
                    min = sp[mid];              
                    found = true;               
                    r = mid - 1;
                }
                else {
                    l = mid + 1;
                }
            }
            result[i] = found ? map[min] : -1;
        }
        return result;
    }
}
/*
If we are not allowed to use TreeMap:
Sort starts
For each end, find leftmost start using binary search
To get the original index, we need a map
*/
public class Solution {
    public int[] FindRightInterval(int[][] intervals) {
          
        Dictionary<int, int> map = new Dictionary<int, int>();
        List<int> starts = new List<int>();
        for (int i = 0; i < intervals.Length; i++) {
        map[intervals[i][0]] = i;
        starts.Add(intervals[i][0]);
        }

       
        starts.Sort((x, y) => (x.CompareTo(y)));
         int[] res = new int[intervals.Length];
         for (int i = 0; i < intervals.Length; i++) {
             int end = intervals[i][intervals[i].Length - 1];
             int start = binarySearch(starts, end);
             if (start < end) {
                 res[i] = -1;
             } else {
                 res[i] = map[start];
             }
         }
         return res;
}

public int binarySearch(List<int> list, int x) {
        int l = 0, r = list.Count - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (list[m] < x) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return list[l];

    }
}

// Day 12
// 81. Search in Rotated Sorted Array II
/*
Solution: Binary search or divide and conquer
If current range is ordered, use binary search, Otherwise, divide and conquer.

Time complexity: O(logn) best, O(n) worst
Space complexity: O(logn)
*/
public class Solution {
    public bool Search(int[] nums, int target) {
        return search(nums, 0, nums.Length - 1, target);
  }
    private bool search(int[] A, int l, int r, int target) {
        if (l > r) return false;
        if (l == r) return target == A[l];
    
        int mid = l + (r - l) / 2;
    
        if (A[l] < A[mid] && A[mid] < A[r])
        return (target <= A[mid]) ? search(A, l, mid, target) : search(A, mid + 1, r, target);
        else
        return search(A, l, mid, target) || search(A, mid + 1, r, target);
    }
}

// 162. Find Peak Element
/*
Solution: Binary Search
Time complexity: O(logn)
Space complexity: O(1)
*/
public class Solution {
    public int FindPeakElement(int[] nums) {
        int l = 0, r = nums.Length - 1; // preventing OOB
        while (l < r) {
          int m = l + (r - l) / 2;
          // Find the first m s.t. num[m] > num[m + 1]
          if (nums[m] > nums[m + 1])
            r = m;
          else
            l = m + 1;
        }
        return l;
    }
}

// Day 13
// 154. Find Minimum in Rotated Sorted Array II
/*
Divide and conquer
Time complexity:
Average: O(logn)
Worst: O(n
*/
public class Solution {
    public int FindMin(int[] nums) {
        return findMin(nums, 0, nums.Length-1);
    }
    
    int findMin(int[] num, int l, int r)
    {
        // One or two elements, solve it directly
        if (l+1 >= r) return
            Math.Min(num[l], num[r]);
        
        // Sorted
        if (num[l] < num[r])
            return num[l];
        
        int m = l + (r-l)/2;
        
        // Recursively find the solution
        return Math.Min(findMin(num, l, m - 1), 
                   findMin(num, m, r));
    }
}

/*Binary Search
First, we take
low (lo) as 0
high (hi) as nums.length-1

By default, if nums[lo]<nums[hi] then we return nums[lo] because the array was never rotated, or is rotated n times

After entering while loop, we check
if nums[mid] > nums[hi] => lo = mid + 1 because the minimum element is in the right half of the array
else if nums[mid] < nums[hi] => hi = mid because the minimum element is in the left half of the array
else => hi-- dealing with duplicate values
then we return nums[hi]
*/
public class Solution {
    public int FindMin(int[] nums) {
       int l = 0, r = nums.Length - 1;
        
        if (nums[l] < nums[r]) return nums[l];
        
        while (l < r) {
            int m = l + (r - l) / 2;
            
            if (nums[m] > nums[r]) {
                l = m + 1;
            } else if (nums[m] < nums[r]) {
                r = m;
            } else { //nums[mid]=nums[r] no idea, but we can eliminate nums[r];
                r--;
            }
        }
        
        return nums[r];
    }
}
// 528. Random Pick with Weight
/*
Solution: Binary Search
Crate a cumulative weight array, random sample a “weight”, do a binary search to see which bucket that weight falls in.
e.g. w = [2, 3, 1, 4], sum = [2, 5, 6, 10]
sample 3 => index = 1
sample 7 => index = 3

Time complexity: Init: O(n) Pick: O(logn)
Space complexity: O(n)
*/
public class Solution {
    
    Random random;
    int[] wSums;

    public Solution(int[] w) {
        this.random = new Random();
        for(int i=1; i<w.Length; ++i)
            w[i] += w[i-1];
        this.wSums = w;
    }
    
    public int PickIndex() {
        int len = wSums.Length;
        int idx = random.Next(wSums[len-1]) + 1;
        int l = 0, r = len - 1;
        // search position 
        while(l < r){
            int m = l + (r - l)/2;
            if(wSums[m] == idx)
                return m;
            else if(wSums[m] < idx)
                l = m + 1;
            else
                r = m;
        }
        return l;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(w);
 * int param_1 = obj.PickIndex();
 */

public class Solution {
    
    Random random;
    int[] wSums;

    public Solution(int[] w) {
        this.random = new Random();
        for(int i=1; i<w.Length; ++i)
            w[i] += w[i-1];
        this.wSums = w;
    }
    
    public int PickIndex() {
        int len = wSums.Length;
        int idx = random.Next(wSums[len-1]) + 1;
        int i = Array.BinarySearch(wSums, idx);//Binary Search
        return i >= 0 ? i : -i-1;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(w);
 * int param_1 = obj.PickIndex();
 */

// Day 14
// 1508. Range Sum of Sorted Subarray Sums
/*
Solution : Brute Force
Find sums of all the subarrays and sort the values.

Time complexity: O(n^2logn)
Space complexity: O(n^2)
*/
public class Solution {
    public int RangeSum(int[] nums, int n, int left, int right) {
        int kMod = (int)(1000000000 + 7);
        int[] sums = new int[n * (n + 1) / 2];
        int idx = 0;
        for (int i = 0; i < n; ++i)
          for (int j = i, sum = 0; j < n; ++j, ++idx)
            sums[idx] = sum += nums[j];
        Array.Sort(sums);
        int ans = 0;
        for (int i = left; i <= right; ++i)
          ans = (ans + sums[i - 1]) % kMod;
        return ans;
    }
}
/*
Solution : Binary Search + Sliding Window

Use binary search to find S s.t. that there are at least k subarrys have sum <= S.
Given S, we can use sliding window to count how many subarrays have sum <= S and their total sum.
ans = sums_of_first(right) – sums_of_first(left – 1).

Time complexity: O(n * log(sum(nums))
Space complexity: O(n)
*/
public class Solution {
    int[] A, B, nums;
    int n;

    public int RangeSum(int[] nums, int n, int left, int right) {
    int mod = (int) 1e9 + 7;
        this.n = n;
        this.nums = nums;
        A = new int[n + 1];
        B = new int[n + 1];
        for (int i = 0; i < n; i++) {
            A[i + 1] = A[i] + nums[i];
            B[i + 1] = B[i] + A[i + 1];
        }
        return (int) ((sumKsums(right) - sumKsums(left - 1)) % mod);
    }

    long sumKsums(int k) {
        int score = kthSum(k);
        long sum = 0;
        for (int l = 0, r = 0; r < n; r++) {
            while (score + A[l] < A[r + 1]) {
                l++;
            }
            sum += A[r + 1] * (r - l + 1) - (l == 0 ? B[r] : B[r] - B[l - 1]);
        }
        return sum - (countUnderScore(score) - k) * score;
    }

    int kthSum(int k) {
        int l = 0, r = A[n];
        while (l < r) {
            int m = (l + r) >> 1;
            if (countUnderScore(m) < k) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return l;
    }

    int countUnderScore(int score) {
        int count = 0;
        for (int l = 0, r = 0; r < n; r++) {
            while (score + A[l] < A[r + 1]) {
                l++;
            }
            count += r - l + 1;
        }
        return count;
    
    }
}

//Prefix sum
public class Solution {
    public int RangeSum(int[] nums, int n, int left, int right) {
    long res = 0, min = long.MaxValue, mod = 1_000_000_007, sum = 0;
        List<long> sums = new List<long>(), pSum = new List<long>();  // sums - all sums of subarrays, pSum - prefix sums;
        pSum.Add(0L);
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            pSum.Add(sum);
            for (int j = 0; j < pSum.Count - 1; j++) sums.Add(sum - pSum[j]);
        }
        sums.Sort();
        while (left <= right) res = (res + sums[left++ - 1]) % mod;
        return (int) res;
    }
}

// 1574. Shortest Subarray to be Removed to Make Array Sorted
/*
Solution: Two Pointers
Find the right most j such that arr[j – 1] > arr[j], 
if not found which means the entire array is sorted return 0. 
Then we have a non-descending subarray arr[j~n-1].

We maintain two pointers i, j, such that arr[0~i] is non-descending and arr[i] <= arr[j] 
which means we can remove arr[i+1~j-1] to get a non-descending array. 
Number of elements to remove is j – i – 1 .

Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public int FindLengthOfShortestSubarray(int[] arr) {
        int n = arr.Length;
        int j = n - 1;
        while (j > 0 && arr[j - 1] <= arr[j]) --j;
        if (j == 0) return 0;
        int ans = j; // remove arr[0~j-1]
        for (int i = 0; i < n; ++i) {
          if (i > 0 && arr[i - 1] > arr[i]) break;
          while (j < n && arr[i] > arr[j]) ++j;      
          // arr[i] <= arr[j], remove arr[i + 1 ~ j - 1]
          ans = Math.Min(ans, j - i - 1);
        }
        return ans;
    }
}

// Day 15
// 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
/*
Solution : Bounded Search
Time complexity: O(m*n + min(m,n))
*/
public class Solution {
    public int MaxSideLength(int[][] mat, int threshold) {
   int m = mat.Length;
    int n = mat[0].Length;
    
    int[][] dp = new int[m + 1][];
    for (int i = 0; i < m+1; ++i)
        dp[i] = new int[n + 1];
    for (int y = 1; y <= m; ++y)
      for (int x = 1; x <= n; ++x)
        dp[y][x] = dp[y][x - 1] + dp[y - 1][x]  - dp[y - 1][x - 1] + mat[y - 1][x - 1];
    
    int rangeSum (int x1, int y1, int x2, int y2) {
      return dp[y2][x2] - dp[y2][x1 - 1] - dp[y1 - 1][x2] + dp[y1 - 1][x1 - 1];
    };
    
    int ans = 0;
    for (int y = 1; y <= m; ++y)
      for (int x = 1; x <= n; ++x)
        for (int k = ans; y + k <= m && x + k <= n; ++k) {
          if (rangeSum(x, y, x + k, y + k) > threshold) break;
          ans = Math.Max(ans, k + 1);
        }
    return ans;  
    }
}
/*Solution : Binary Search
Search for the smallest size k that is greater than the threshold, ans = k - 1.*/
public class Solution {
    public int MaxSideLength(int[][] mat, int threshold) {
   int m = mat.Length;
    int n = mat[0].Length;
    
    int[][] dp = new int[m + 1][];
    for (int i = 0; i < m+1; ++i)
        dp[i] = new int[n + 1];
    for (int y = 1; y <= m; ++y)
      for (int x = 1; x <= n; ++x)
        dp[y][x] = dp[y][x - 1] + dp[y - 1][x]  - dp[y - 1][x - 1] + mat[y - 1][x - 1];
    
    int rangeSum(int x1, int y1, int x2, int y2) {
      return dp[y2][x2] - dp[y2][x1 - 1] - dp[y1 - 1][x2] + dp[y1 - 1][x1 - 1];
    };
    
    int ans = 0;
    for (int y = 1; y <= m; ++y)
      for (int x = 1; x <= n; ++x) {
        int l = 0;
        int r = Math.Min(m - y, n - x) + 1;
        while (l < r) {
          int mid = l + (r - l) / 2;
          // Find smllest l that > threshold, ans = (l + 1) - 1
          if (rangeSum(x, y, x + mid, y + mid) > threshold)
            r = mid;
          else
            l = mid + 1;
        }
        ans = Math.Max(ans, (l + 1) - 1);
      }
    return ans;  
    }
}
/*
Solution: DP + Brute Force
Precompute the sums of sub-matrixes whose left-top corner is at (0,0).
Try all possible left-top corner and sizes.

Time complexity: O(m*n*min(m,n))
Space complexity: O(m*n)
*/
public class Solution {
    public int MaxSideLength(int[][] mat, int threshold) {
    int m = mat.Length;
    int n = mat[0].Length;
    
    int[][] dp = new int[m + 1][];
    for (int i = 0; i < m+1; ++i)
        dp[i] = new int[n + 1];
    for (int y = 1; y <= m; ++y)
      for (int x = 1; x <= n; ++x)
        dp[y][x] = dp[y][x - 1] + dp[y - 1][x]  - dp[y - 1][x - 1] + mat[y - 1][x - 1];
    
    int rangeSum(int x1, int y1, int x2, int y2) {
      return dp[y2][x2] - dp[y2][x1 - 1] - dp[y1 - 1][x2] + dp[y1 - 1][x1 - 1];
    };
    
    int ans = 0;
    for (int y = 1; y <= m; ++y)
      for (int x = 1; x <= n; ++x)
        for (int k = 0; y + k <= m && x + k <= n; ++k) {
          if (rangeSum(x, y, x + k, y + k) > threshold) break;
          ans = Math.Max(ans, k + 1);
        }
    return ans;  
    }
}

// 1498. Number of Subsequences That Satisfy the Given Sum Condition
/*  Two Sum
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
(O(N) space for java and c++ can be save anyway)
*/
public class Solution {
    public int NumSubseq(int[] nums, int target) {
        Array.Sort(nums);
        int res = 0, n = nums.Length, l = 0, r = n - 1, mod = (int)1e9 + 7;
        int[] pows = new int[n];
        pows[0] = 1;
        for (int i = 1 ; i < n ; ++i)
            pows[i] = pows[i - 1] * 2 % mod;
        while (l <= r) {
            if (nums[l] + nums[r] > target) {
                r--;
            } else {
                res = (res + pows[r - l++]) % mod;
            }
        }
        return res;
    }
}

/*
Solution: Two Pointers
Since order of the elements in the subsequence doesn’t matter, we can sort the input array.
Very similar to two sum, we use two pointers (i, j) to maintain a window, s.t. nums[i] +nums[j] <= target.
Then fix nums[i], any subset of (nums[i+1~j]) gives us a valid subsequence, thus we have 2^(j-(i+1)+1) = 2^(j-i) valid subsequence for window (i, j).

Time complexity: O(nlogn) // Sort
Space complexity: O(n) // need to precompute 2^n % kMod.
*/
public class Solution {
    public int NumSubseq(int[] nums, int target) {

        int kMod = 1000000000 + 7;
        int n = nums.Length;
        int[] p = new int[n + 1]; p[0] = 1;
    for (int i = 1; i <= n; ++i) 
      p[i] = (p[i - 1] << 1) % kMod;
     Array.Sort(nums);
    int ans = 0;
    for (int i = 0, j = n - 1; i <= j; ++i) {
      while (i <= j && nums[i] + nums[j] > target) --j;
      if (i > j) continue;
      // In subarray nums[i~j]:
      // min = nums[i], max = nums[j]
      // nums[i] + nums[j] <= target
      // {nums[i], (j - i - 1 + 1 values)}
      // Any subset of the right part gives a valid subsequence 
      // in the original array. And There are 2^(j - i) ones.
      ans = (ans + p[j - i]) % kMod;
    }
    return ans;
    }
}
// Day 16
// 981. Time Based Key-Value Store
public class Data {
    public String val;
    public int time;
    public Data(String val, int time) {
        this.val = val;
        this.time = time;
    }
}
public class TimeMap {
    
    /** Initialize your data structure here. */
    Dictionary<String, List<Data>> map;
    public TimeMap() {
        map = new Dictionary<String, List<Data>>();
    }
    
    public void Set(string key, string value, int timestamp) {
        if (!map.ContainsKey(key)) map[key] = new List<Data>();
        map[key].Add(new Data(value, timestamp));
    }
    
    public string Get(string key, int timestamp) {
     if (!map.ContainsKey(key)) return "";
        return binarySearch(map[key], timestamp);
    }
    
    public string binarySearch(List<Data> list, int time) {
        int low = 0, high = list.Count - 1;
        while (low < high) {
            int mid = low + (high - low)/2;
            if (list[mid].time == time) return list[mid].val;
            if (list[mid].time < time) {
                if (list[mid+1].time > time) return list[mid].val;
                low = mid + 1;
            }
            else high = mid -1;
        }
        return list[low].time <= time ? list[low].val : "";
    }
}

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap obj = new TimeMap();
 * obj.Set(key,value,timestamp);
 * string param_2 = obj.Get(key,timestamp);
 */

// 1300. Sum of Mutated Array Closest to Target
/*
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
*/
public class Solution {
    public int FindBestValue(int[] arr, int target) {
        Array.Sort(arr);
        int n = arr.Length, i = 0;
        while (i < n && target > arr[i] * (n - i)) {
            target -= arr[i++];
        }
        if (i == n) return arr[n - 1];
        int res = target / (n - i);
        if (target - res * (n - i) > (res + 1) * (n - i) - target)
            res++;
        return res;
    }
}

// Day 17
// 1802. Maximum Value at a Given Index in a Bounded Array
/* Binary Search
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
Space O(1)*/
public class Solution {
    public int MaxValue(int n, int index, int maxSum) {
        maxSum -= n;
        int left = 0, right = maxSum, mid;
        while (left < right) {
            mid = (left + right + 1) / 2;
            if (test(n, index, mid) <= maxSum)
                left = mid;
            else
                right = mid - 1;
        }
        return left + 1;
    }
    
    private long test(int n, int index, int a) {
        int b = Math.Max(a - index, 0);
        long res = (long)(a + b) * (a - b + 1) / 2;
        b = Math.Max(a - ((n - 1) - index), 0);
        res += (long)(a + b) * (a - b + 1) / 2;
        return res - a;
    
    }
}

// 1901. Find a Peak Element II
public class Solution {
    public int[] FindPeakGrid(int[][] mat) {
        int top = 0;
        int bottom = mat.Length -1;
        while (bottom > top){
            int mid = (top + bottom) / 2;
            if (mat[mid].Max() > mat[mid+1].Max()){
                 bottom = mid;
            }
            else{
                top = mid+1;
            }
                
        }
        return new int[2]{bottom,Array.IndexOf(mat[bottom], mat[bottom].Max())};
    }
}

// Day 18
// 1146. Snapshot Array
/*Binary Search
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
*/
public class SnapshotArray {
    List<int[]>[] record;
    int sid;

    public SnapshotArray(int length) {
        record = new List<int[]>[length];
        sid = 0;
        for (int i = 0; i < length; i++) {
            record[i] = new List<int[]>();
            record[i].Add(new int[]{0, 0});
        }
    }
    
    public void Set(int index, int val) {
        if (record[index][record[index].Count - 1][0] == sid) {
            record[index][record[index].Count - 1][1] = val;
        } else 
            record[index].Add(new int[]{sid, val});
    }
    
    public int Snap() {
        return sid++;
    }
    
    public int Get(int index, int snap_id) {
        int idx = record[index].BinarySearch(new int[]{snap_id, 0}, 
                                          Comparer<int[]>.Create((a, b) => a[0] - b[0] ));
        if (idx < 0) idx = - idx - 2;
        return record[index][idx][1];
    }
}

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray obj = new SnapshotArray(length);
 * obj.Set(index,val);
 * int param_2 = obj.Snap();
 * int param_3 = obj.Get(index,snap_id);
 */

// 1488. Avoid Flood in The City
/*
Solution: Binary Search
Store the days we can dry a lake in a treeset.
Store the last day when a lake becomes full in a hashtable.
Whenever we encounter a full lake, try to find the first available day that we can dry it. If no such day, return no answer.

Time complexity: O(nlogn)
Space complexity: O(n)
*/
public class Solution {
    public int[] AvoidFlood(int[] rains) {
        int n = rains.Length;
        int[] ans = new int[n]; Array.Fill(ans, -1);
        Dictionary<int, int> full = new Dictionary<int,int>(); // lake -> day
        List<int> dry = new List<int>(); // days we can dry lakes.
        for (int i = 0; i < n; ++i) {
          int lake = rains[i];
          if (lake > 0) {
            if (full.ContainsKey(lake)) {
              // Find the first day we can dry it. 
                //List<T>.BinarySearch use like this as upperbound in C++
                int it = dry.BinarySearch(full[lake]+1); 
                // have to make sure the below two lines order like this
                if(it < 0) it = ~it;
              if (it == dry.Count) return new int[0];
              ans[dry[it]] = lake;
              dry.RemoveAt(it);
            }
            full[lake] = i;
          } 
            else {
            dry.Add(i);
            ans[i] = 1;
          }
        }
        return ans;
    }
}

public class Solution {
    public int[] AvoidFlood(int[] rains) {
        var zeros = new List<int>();
        var prev = new Dictionary<int, int>();
        var result = Enumerable.Repeat(-1, rains.Length).ToArray();
        
        for(int i = 0; i < rains.Length; i++)
        {
            var lake = rains[i];
            if(rains[i] == 0)
            {
                zeros.Add(i);
                result[i] = 1;
            }
            else
            {
                if(!prev.ContainsKey(lake))
                    prev[lake] = i;
                else
                {
                    var prevIndex =  prev[lake];
                    var index = zeros.BinarySearch(prevIndex + 1);
                    if(index < 0)
                        index = ~index;
                    if(index == zeros.Count)
                        return new int[0];
                    prev[lake] = i;
                    result[zeros[index]] = lake;
                    zeros.RemoveAt(index);
                }
            }
        }
        
        return result;
    
    }
}
// Day 19
// 1562. Find Latest Group of Size M
/*
Solution: Hashtable
Similar to LC 128

Time complexity: O(n)
Space complexity: O(n)
*/
public class Solution {
    public int FindLatestStep(int[] arr, int m) {
        int n = arr.Length;
    int[] len = new int[n + 2];
    int[] counts = new int[n + 2];
    int ans = -1;
    for (int i = 0; i < n; ++i) {
      int x = arr[i];      
      int l = len[x - 1];
      int r = len[x + 1];
      int t = 1 + l + r;  
      len[x - l] = len[x + r] = t;
      --counts[l];
      --counts[r];
      ++counts[t];
      if (counts[m] > 0) ans = i + 1;
    }
    return ans;
    }
}
/*
Solution suggests removing the counter.
*/
public class Solution {
    public int FindLatestStep(int[] arr, int m) {
        int res = -1, n = arr.Length;
        if (n == m) return n;
        int[] length = new int[n + 2];
        for (int i = 0; i < n; ++i) {
            int a = arr[i], left = length[a - 1], right = length[a + 1];
            length[a - left] = length[a + right] = left + right + 1;
            if (left == m || right == m)
                res = i;
        }
        return res;
    }
}
/* Count the Length of Groups, O(N)
Explanation
When we set bit a, where a = A[i],
we check the length of group on the left length[a - 1]
also the length of group on the right length[a + 1].
Then we update length[a - left], length[a + right] to left + right + 1.

Note that the length value is updated on the leftmost and the rightmost bit of the group.
The length value inside the group may be out dated.

As we do this, we also update the count of length.
If count[m] > 0, we update res to current step index i + 1.

Complexity
Time O(N)
Space O(N)

Solution : Count all lengths
*/
public class Solution {
    public int FindLatestStep(int[] arr, int m) {
        int res = -1, n = arr.Length;
        int[] length = new int[n + 2], count = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            int a = arr[i], left = length[a - 1], right = length[a + 1];
            length[a] = length[a - left] = length[a + right] = left + right + 1;
            count[left]--;
            count[right]--;
            count[length[a]]++;
            if (count[m] > 0)
                res = i + 1;
        }
        return res;
    }
}
// 1648. Sell Diminishing-Valued Colored Balls
/*
Solution: Greedy
Sort the colors by # of balls in descending order.
e.g. 3 7 5 1 => 7 5 3 1
Sell the color with largest number of balls until it has the same number of balls of next color
7 5 3 1 => 6 5 3 1 => 5 5 3 1 # value = 7 + 6 = 13
5 5 3 1 => 4 4 3 1 => 3 3 3 1 # value = 13 + (5 + 4) * 2 = 31
3 3 3 1 => 2 2 2 1 => 1 1 1 1 # value = 31 + (3 + 2) * 3 = 46
1 1 1 1 => 0 0 0 0 # value = 46 + 1 * 4 = 50
Need to handle the case if orders < total balls…
Time complexity: O(nlogn)
Space complexity: O(1)
*/
public class Solution {
    public int MaxProfit(int[] inventory, int orders) {
        int kMod = 1000000000 + 7;
        int n = inventory.Length;
        inventory= inventory.OrderByDescending(c => c).ToArray(); ;
        long cur = inventory[0];
        long ans = 0;
        int c = 0;
        while (orders > 0) {      
        while (c < n && inventory[c] == cur) ++c;
        int nxt = c == n ? 0 : inventory[c];      
        int count = (int)Math.Min((long)(orders), c * (cur - nxt));
        int t = (int)(cur - nxt);
        int r = 0;
        if (orders < c * (cur - nxt)) {
            t = orders / c;
            r = orders % c;
        }
        ans = (ans + (cur + cur - t + 1) * t / 2 * c + (cur - t) * r) % kMod;
        orders -= count;
        cur = nxt;
        }
        return (int)ans;
    }
}

// Day 20
// 1201. Ugly Number III
/*Solution: Binary Search

Number of ugly numbers that are <= m are:

m / a + m / b + m / c – (m / LCM(a,b) + m / LCM(a, c) + m / LCM(b, c) + m / LCM(a, LCM(b, c))

Time complexity: O(logn)
Space complexity: O(1)*/
public class Solution {
    public int NthUglyNumber(int n, int a, int b, int c) {
        long l = 1;
    long r = Int32.MaxValue;
    long ab = lcm(a, b);
    long ac = lcm(a, c);
    long bc = lcm(b, c);
    long abc = lcm(a, bc);
    while (l < r) {
      long m = l + (r - l) / 2;
      long k = m / a + m / b + m / c - m / ab - m / ac - m / bc + m / abc;      
      if (k >= n) r = m;
      else l = m + 1;
    }
    return (int)l;
    }
    public long gcd(long a, long b) {
        if (a == 0) return b;
        return gcd(b % a, a);
    }
    public long lcm(long a, long b) {
        return a * b / gcd(a, b);
    }
}

// 911. Online Election
/*
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
*/
public class TopVotedCandidate {

    Dictionary<int, int> m = new Dictionary<int, int>();
    int[] time;
    public TopVotedCandidate(int[] persons, int[] times) {
         int n = persons.Length, lead = -1;
        Dictionary<int, int> count = new Dictionary<int, int>();
        time = times;
        for (int i = 0; i < n; ++i) {
            count[persons[i]] = count.GetValueOrDefault(persons[i], 0) + 1;
            if (i == 0 || count[persons[i]] >= count[lead]) lead = persons[i];
            m[times[i]] = lead;
        }
    }
    
    public int Q(int t) {
        int i = Array.BinarySearch(time, t);
        return i < 0 ? m[time[-i-2]] : m[time[i]];
    }
}

/**
 * Your TopVotedCandidate object will be instantiated and called as such:
 * TopVotedCandidate obj = new TopVotedCandidate(persons, times);
 * int param_1 = obj.Q(t);
 */