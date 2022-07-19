// Day 1 Basic Data Type
// 1523. Count Odd Numbers in an Interval Range
public class Solution {
    public int CountOdds(int low, int high) {
        return (high + 1) / 2 - low / 2;
    }
}

// 1491. Average Salary Excluding the Minimum and Maximum Salary
public class Solution {
    public double Average(int[] salary) {
        int min = salary.Min();
        int max = salary.Max();
        double sum = (double)salary.Sum() - (max + min);
        
        return sum / (salary.Length - 2);
    }
}

// Day 2 Operator
// 191. Number of 1 Bits
public class Solution {
    public int HammingWeight(uint n) {
        int count = 0;
    
        while (n>0) {
            n &= (n - 1);
            count++;
        }

        return count;
    }
}

// 1281. Subtract the Product and Sum of Digits of an Integer
public class Solution {
    public int SubtractProductAndSum(int n) {
        int sum = 0, prod = 1;
        while(n>0)
        {
            sum += n%10;
            prod *= n%10;
            n = n/10;
        }
        return prod - sum;
    }
}

// Day 3 Conditional Statements
// 976. Largest Perimeter Triangle
public class Solution {
    public int LargestPerimeter(int[] nums) {
        Array.Sort(nums);
        for (int i = nums.Length - 1; i > 1; i--)
            if (nums[i] < nums[i - 1] + nums[i - 2])
                return nums[i] + nums[i - 1] + nums[i - 2];
        return 0;
    }
}
// 1779. Find Nearest Point That Has the Same X or Y Coordinate
public class Solution {
    public int NearestValidPoint(int x, int y, int[][] points) {
        int index = -1, smallest = int.MaxValue;
        for (int i = 0;i < points.Length; i++) {
            if ((x == points[i][0] || y == points[i][1] ) && Math.Abs(x - points[i][0]) + Math.Abs( y - points[i][1])<smallest) {
                smallest = Math.Abs(x - points[i][0])+Math.Abs( y - points[i][1]);
                index = i;
            }
        }
        return index;
    }
}

// Day 4 Loop
// 1822. Sign of the Product of an Array
// flip the count whenever encounter a negative number
public class Solution {
    public int ArraySign(int[] nums) {
        int sign = 1; 
        foreach (int n in nums) {
            if (n == 0) {
                return 0; 
            } 
			if (n < 0) {
                sign = -sign; 
            }
        }
        return sign; 
    }
}
// 1502. Can Make Arithmetic Progression From Sequence
public class Solution {
    public bool CanMakeArithmeticProgression(int[] arr) {
        List<int> seen = new List<int>();
        int mi = Int32.MaxValue, mx = Int32.MinValue, n = arr.Length;
        foreach (int a in arr) {
            mi = Math.Min(mi, a);
            mx = Math.Max(mx, a);
            seen.Add(a);
        }
        int diff = mx - mi;
        if (diff % (n - 1) != 0) {
            return false;
        }
        diff /= n - 1;
        while (--n > 0) {
            if (!seen.Contains(mi)) {
                return false;
            }
            mi += diff;
        }
        return true;
    }
}
// 202. Happy Number
public class Solution {
    
    private static List<int> cycleMembers =
        new List<int>(){4, 16, 37, 58, 89, 145, 42, 20};
    public int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }


    public bool IsHappy(int n) {
        while (n != 1 && !cycleMembers.Contains(n)) {
            n = getNext(n);
        }
        return n == 1;
    }
}

// 1790. Check if One String Swap Can Make Strings Equal
public class Solution {
    public bool AreAlmostEqual(string s1, string s2) {
        List<int> idx = new List<int>();
    for (int i = 0; i < s1.Length && idx.Count <= 2; ++i)
      if (s1[i] != s2[i]) idx.Add(i);
    if (idx.Count == 0) return true;
    if (idx.Count != 2) return false;
    return s1[idx[0]] == s2[idx[1]] && s1[idx[1]] == s2[idx[0]];
    }
}
// Day 5 Function
// 589. N-ary Tree Preorder Traversal
/*
// Definition for a Node.
public class Node {
    public int val;
    public IList<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,IList<Node> _children) {
        val = _val;
        children = _children;
    }
}
*/
//Solution : Iterative
public class Solution {
    public IList<int> Preorder(Node root) {
        if (root == null) return new List<int>();
        List<int> ans = new List<int>();
        Stack<Node> s = new Stack<Node>();
        s.Push(root);
        while (s.Count != 0) {
          Node node = s.Peek(); s.Pop();
          ans.Add(node.val);
          for (int i = node.children.Count - 1 ; i>= 0 ; i--)
            s.Push(node.children[i]);
        }
        return ans;
  }
}

/*
// Definition for a Node.
public class Node {
    public int val;
    public IList<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,IList<Node> _children) {
        val = _val;
        children = _children;
    }
}
*/
// Solution : Recursive
public class Solution {
    public IList<int> Preorder(Node root) {
        List<int> ans = new List<int>();
    preorder(root, ans);
    return ans;
  }
private void preorder(Node root, List<int> ans) {
    if (root == null) return;
    ans.Add(root.val);
    foreach (Node child in root.children)
      preorder(child, ans);
  }
}
// 496. Next Greater Element I
/*
Key observation:
Suppose we have a decreasing sequence followed by a greater number
For example [5, 4, 3, 2, 1, 6] then the greater number 6 is the next greater element for all previous numbers in the sequence

We use a stack to keep a decreasing sub-sequence, whenever we see a number x greater than stack.peek() we pop all elements less than x and for all the popped ones, their next greater element is x
For example [9, 8, 7, 3, 2, 1, 6]
The stack will first contain [9, 8, 7, 3, 2, 1] and then we see 6 which is greater than 1 so we pop 1 2 3 whose next greater element should be 6
*/
public class Solution {
    public int[] NextGreaterElement(int[] nums1, int[] nums2) {
        Dictionary<int, int> map = new Dictionary<int,int>(); // map from x to next greater element of x
        Stack<int> stack = new Stack<int>();
        foreach (int num in nums2) {
            while (stack.Count != 0 && stack.Peek() < num)
                map[stack.Pop()] = num;
            stack.Push(num);
        }   
        for (int i = 0; i < nums1.Length; i++)
            nums1[i] = map.GetValueOrDefault(nums1[i], -1);
        return nums1;
    }
}

// 1232. Check If It Is a Straight Line
/*
The slope for a line through any 2 points (x0, y0) and (x1, y1) is (y1 - y0) / (x1 - x0); Therefore, for any given 3 points (denote the 3rd point as (x, y)), if they are in a straight line, the slopes of the lines from the 3rd point to the 2nd point and the 2nd point to the 1st point must be equal:

(y - y1) / (x - x1) = (y1 - y0) / (x1 - x0)
In order to avoid being divided by 0, use multiplication form:

(x1 - x0) * (y - y1) = (x - x1) * (y1 - y0) =>
dx * (y - y1) = dy * (x - x1), where dx = x1 - x0 and dy = y1 - y0
Now imagine connecting the 2nd points respectively with others one by one, Check if all of the slopes are equal.
*/
public class Solution {
    public bool CheckStraightLine(int[][] coordinates) {
        int x0 = coordinates[0][0], y0 = coordinates[0][1], 
            x1 = coordinates[1][0], y1 = coordinates[1][1];
        int dx = x1 - x0, dy = y1 - y0;
        foreach (int[] co in coordinates) {
            int x = co[0], y = co[1];
            if (dx * (y - y1) != dy * (x - x1))
                return false;
        }
        return true;
    }
}
// Day 6 Array
// 1588. Sum of All Odd Length Subarrays
/*
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
*/
/*
Solution 2: Math
Count how many times arr[i] can be in of an odd length subarray
we chose the start, which can be 0, 1, 2, … i, i + 1 choices
we chose the end, which can be i, i + 1, … n – 1, n – i choices
Among those 1/2 are odd length.
So there will be upper((i + 1) * (n – i) / 2) odd length subarrays contain arr[i]

ans = sum(((i + 1) * (n – i) + 1) / 2 * arr[i] for in range(n))

Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public int SumOddLengthSubarrays(int[] arr) {
        int n = arr.Length;    
    int ans = 0;
    for (int i = 0; i < n; ++i)
      ans += ((i + 1) * (n - i) + 1) / 2 * arr[i];
    return ans;
    }
}
/*
Solution 1: Running Prefix Sum
Reduce the time complexity to O(n^2)
*/
public class Solution {
    public int SumOddLengthSubarrays(int[] arr) {
        int n = arr.Length;    
    int ans = 0;
    for (int i = 0, s = 0; i < n; ++i, s = 0)
      for (int j = i; j < n; ++j) {
        s += arr[j];
        ans += s * ((j - i + 1) & 1);
      }    
    return ans;
    }
}
// 283. Move Zeroes
public class Solution {
    public void MoveZeroes(int[] nums) {
        int slow = 0;
        for(int i=0;i<nums.Length;i++)
        {
            if(nums[i]!=0)
            {
                tmp= nums[slow];
                nums[slow] = nums[i];
                nums[i] = tmp;
                slow++;
            }
        }
    }
}
// 1672. Richest Customer Wealth
/* Solution: Sum each row up
Time complexity: O(mn)
Space complexity: O(1)*/
public class Solution {
    public int MaximumWealth(int[][] accounts) {
        int ans = 0;
    foreach (int[] row in accounts)
      ans = Math.Max(ans, row.Aggregate(0, (s, n) => s + n));
    return ans;
    }
}

public class Solution {
    public int MaximumWealth(int[][] accounts) {
        int res = 0;
        for(int i =0;i<accounts.Length;i++){
            int temp = 0;
            for(int j = 0;j<accounts[i].Length;j++){
                temp+=accounts[i][j];
            }
            res = Math.Max(res,temp);
        }
        return res;
    }
}
// Day 7 Array
// 1572. Matrix Diagonal Sum
/*
Solution: Brute Force
Note: if n is odd, be careful not to double count the center one.

Time complexity: O(n)
Space complexity: O(1)*/
public class Solution {
    public int DiagonalSum(int[][] mat) {
        int n = mat.Length;
        int ans = 0;
        for (int i = 0; i < n; ++i)
          ans += mat[i][i] + mat[i][n - i - 1];
        if ( n % 2 == 1) ans -= mat[n / 2][n / 2];
        return ans;
       
    }
}
// 566. Reshape the Matrix
public class Solution {
    public int[][] MatrixReshape(int[][] mat, int r, int c) {
        int n = mat.Length, m = mat[0].Length;
        if(r*c != n*m) return mat;
        /*While a true 2D array would have m rows of n elements each, 
        a jagged array could have m rows each having different numbers of elements.*/
        int[][] result = new int[r][];
        for(int i = 0; i < r*c;i++)
        {
            if(i%c == 0)
            {
                 result[i/c] = new int[c];
            }
            result[i/c][i%c] = mat[i/m][i%m];
            
        }
        return result;
    }
}

public class Solution {
    public int[][] MatrixReshape(int[][] mat, int r, int c) {
        int m = mat.Length, n = mat[0].Length;
        if (m * n != r * c) return mat;
        
        int[][] result = new int[r][];
        for (int i = 0; i < r; i++) {result[i] = new int[c];}
        int row = 0, col = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[row][col] = mat[i][j];
                col++;
                if (col == c) {
                    col = 0;
                    row++;
                }
            }
        }
        
        return result;
    }
}
// Day 8 String
// 1768. Merge Strings Alternately
/*
Solution: Find the shorter one
Time complexity: O(m+n)
Space complexity: O(1)*/
public class Solution {
    public string MergeAlternately(string word1, string word2) {
        int l1 = word1.Length, l2 = word2.Length;
        string ans = String.Empty;
        for (int i = 0; i < Math.Min(l1, l2); ++i) {
        ans += word1[i]; 
        ans += word2[i];
        }
        ans += l1 > l2 ? word1.Substring(l2) : word2.Substring(l1);    
        return ans;
    }
}
/*
Explanation
Alternatively append the character from w1 and w2 to res.

Complexity
Time O(n + m)
Space O(n + m)

Solution : Two Pointers
*/
public class Solution {
    public string MergeAlternately(string word1, string word2) {
        int n = word1.Length, m = word2.Length, i = 0, j = 0;
        StringBuilder res = new StringBuilder();
        while (i < n || j < m) {
            if (i < word1.Length)
                res.Append(word1[i++]);
            if (j < word2.Length)
                res.Append(word2[j++]);
        }
        return res.ToString();
    }
}

// Solution 2: One Pointer
public class Solution {
    public string MergeAlternately(string word1, string word2) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < word1.Length || i < word2.Length; ++i) {
            if (i < word1.Length)
                res.Append(word1[i]);
            if (i < word2.Length)
                res.Append(word2[i]);
        }
        return res.ToString();
    }
}
// 1678. Goal Parser Interpretation
//Without Regex with StringBuilder
public class Solution {
    public string Interpret(string command) {
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < command.Length; ++i) {
        if (command[i] == 'G') ans.Append("G");
        else if (command[i] == '(') {
            if (command[i + 1] == ')') ans.Append("o");
            else ans.Append("al");
        }
        }
        return ans.ToString();

    }
}
/*
Solution: String
If we encounter ‘(‘ check the next character to determine whether it’s ‘()’ or ‘(al’)

Time complexity: O(n)
Space complexity: O(n)
*/
public class Solution {
    public string Interpret(string command) {
        string ans = String.Empty;
    for (int i = 0; i < command.Length; ++i) {
      if (command[i] == 'G') ans += "G";
      else if (command[i] == '(') {
        if (command[i + 1] == ')') ans += "o";
        else ans += "al";
      }
    }
    return ans;

    }
}
// With Regex
public class Solution {
    public string Interpret(string command) {
        return command.Replace("()", "o").Replace("(al)", "al");

    }
}
// 389. Find the Difference
/*
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
*/
public class Solution {
    public char FindTheDifference(string s, string t) {
        char c = (char)0;
        foreach(char cs in s.ToCharArray()) c ^= cs;
        foreach(char ct in t.ToCharArray()) c ^= ct;
        return c;
    }
}
// Day 9 String
// 709. To Lower Case
/*
Solution
Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public string ToLowerCase(string s) {
        char[] c = s.ToCharArray();
    for (int i = 0; i < c.Length; ++i)
      if (c[i] >= 'A' && c[i] <= 'Z') c[i] = (char)(c[i] - 'A' + 'a');
    return new String(c);
    }
}
// 1309. Decrypt String from Alphabet to Integer Mapping
// Regex
public class Solution {
    public string FreqAlphabets(string s) {
        return s.Replace("10#", "j")
            .Replace("11#", "k")
            .Replace("12#", "l")
            .Replace("13#", "m")
            .Replace("14#", "n")
            .Replace("15#", "o")
            .Replace("16#", "p")
            .Replace("17#", "q")
            .Replace("18#", "r")
            .Replace("19#", "s")
            .Replace("20#", "t")
            .Replace("21#", "u")
            .Replace("22#", "v")
            .Replace("23#", "w")
            .Replace("24#", "x")
            .Replace("25#", "y")
            .Replace("26#", "z")
            .Replace("1", "a")
            .Replace("2", "b")
            .Replace("3", "c")
            .Replace("4", "d")
            .Replace("5", "e")
            .Replace("6", "f")
            .Replace("7", "g")
            .Replace("8", "h")
            .Replace("9", "i");
    }
}
// 953. Verifying an Alien Dictionary
/*Mapping to Normal Order
Explanation
Build a transform mapping from order,
Find all alien words with letters in normal order.

For example, if we have order = "xyz..."
We can map the word "xyz" to "abc" or "123"

Then we check if all words are in sorted order.

Complexity
Time O(NS)
Space O(1)*/
public class Solution {
    int[] mapping = new int[26];
    public bool IsAlienSorted(string[] words, string order) {
       for (int i = 0; i < order.Length; i++)
            mapping[order[i] - 'a'] = i;
        for (int i = 1; i < words.Length; i++)
            if (bigger(words[i - 1], words[i]))
                return false;
        return true;
    }

    bool bigger(String s1, String s2) {
        int n = s1.Length, m = s2.Length;
        for (int i = 0; i < n && i < m; ++i)
            if (s1[i] != s2[i])
                return mapping[s1[i] - 'a'] > mapping[s2[i] - 'a'];
        return n > m; 
    }
}
// Day 10 Linked List & Tree
// 1290. Convert Binary Number in a Linked List to Integer
//Approach : Bit Manipulation
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public int GetDecimalValue(ListNode head) {
        int num = head.val;
        while (head.next != null) {
            num = (num << 1) | head.next.val;
            head = head.next;    
        }
        return num;
    }
}
// Approach : Binary Representation
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public int GetDecimalValue(ListNode head) {
        int num = head.val;
        while (head.next != null) {
            num = num * 2 + head.next.val;
            head = head.next;    
        }
        return num;
    }
}

// 876. Middle of the Linked List
/* Slow and Fast Pointers
Each time, slow go 1 steps while fast go 2 steps.
When fast arrives at the end, slow will arrive right in the middle.*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode MiddleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
// 104. Maximum Depth of Binary Tree
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
    public int MaxDepth(TreeNode root) {
        if (root == null) return 0;
        int l = MaxDepth(root.left);
        int r = MaxDepth(root.right);
        return Math.Max(l, r) + 1;
    }
}
// 404. Sum of Left Leaves
/*Solution: Recursion
Time complexity: O(n)
Space complexity: O(h)*/
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
    public int SumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
    if (root.left != null && root.left.left == null && root.left.right == null)
        return root.left.val + SumOfLeftLeaves(root.right);
    return SumOfLeftLeaves(root.left) + SumOfLeftLeaves(root.right);
    }
}

// Iterative
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
    public int SumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
    Queue<TreeNode> q = new Queue<TreeNode>();
    q.Enqueue(root);
    int sum = 0;
    while (q.Count != 0) {
      TreeNode n = q.Peek();            
      q.Dequeue();
 
      TreeNode l = n.left;
      if (l != null) {
        if (l.left == null && l.right == null)
            sum += l.val;
        else
            q.Enqueue(l);
      }
      if (n.right != null) q.Enqueue(n.right);
    }
    return sum;
    }
}
// Day 11 Containers & Libraries
// 1356. Sort Integers by The Number of 1 Bits
/*
Explanation by @Be_Kind_To_One_Another:
Comment regarding i -> Integer.bitCount(i) * 10000 + i to answer any potential question. 
It's essentially hashing the original numbers into another number generated from the count of bits and then sorting the newly generated numbers. 
so why 10000? simply because of the input range is 0 <= arr[i] <= 10^4.
For instance [0,1,2,3,5,7], becomes something like this [0, 10001, 10002, 20003, 20005, 30007].

0 has 0 number of bits  --> 0 * 10000 + 0 = 0
1,2 have 1 bit set      --> 1 * 10000 + 1 = 10001  &  1 * 10000 + 2 = 10002
3,5 have 2 bits set     --> 2 * 10000 + 3 = 20003  &  2 * 10000 + 5 = 20005
7 has 3 bits set        --> 3 * 10000 + 7 = 30007
In short, the bit length contribution to the value of hash code is always greater than the number itself. 
*/
public class Solution {
    public int[] SortByBits(int[] arr) {
        Array.Sort(arr, (Comparer<int>.Create((a,b) => (numberOfOne(a) * 10000 + a) - (numberOfOne(b) * 10000 + b))));
        return arr;
    }
    public int numberOfOne(int n) {
        int count = 0;
        while (n != 0)
        {
            count++;
            n &= (n - 1);
        }
	    return count;
    }
}

// 232. Implement Queue using Stacks
public class MyQueue {

    public Stack<int> s1 = new Stack<int>();
    public Stack<int> s2 = new Stack<int>();
    public int front;
    
    public MyQueue() {
        
    }
    
    public void Push(int x) {
        if (s1.Count == 0) front = x;
        s1.Push(x);
    }
    
    public int Pop() {
        if(s2.Count == 0)
        {
            while(s1.Count > 0)
            {
                s2.Push(s1.Pop());
            }
        }
        return s2.Pop();
    }
    
    public int Peek() {
        if(s2.Count > 0)
        {
            return s2.Peek();
        }
        return front;
    }
    
    public bool Empty() {
        return s1.Count == 0 && s2.Count == 0;
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.Push(x);
 * int param_2 = obj.Pop();
 * int param_3 = obj.Peek();
 * bool param_4 = obj.Empty();
 */

// 242. Valid Anagram
public class Solution {
    public bool IsAnagram(string s, string t) {
        Dictionary<Char,int> map = new Dictionary<Char,int>();
        if(s.Length!=t.Length){return false;}
        for(int i=0;i<s.Length;i++){
            if (map.ContainsKey(s[i])) map[s[i]]+=1;
            else map[s[i]]=1;
            if (map.ContainsKey(t[i])) map[t[i]]-=1;
            else map[t[i]]=-1;
            
        }
        foreach(int i in map.Values) if(i != 0) return false;
        return true;
    }
}

public class Solution {
    public bool IsAnagram(string s, string t) {
        int[] alphabet = new int[26];
        for (int i = 0; i < s.Length; i++) alphabet[s[i] - 'a']++;
        for (int i = 0; i < t.Length; i++) alphabet[t[i] - 'a']--;
        for (int i =0; i< alphabet.Length; i++) if (alphabet[i] != 0) return false;
        return true;
    }
}
// 217. Contains Duplicate
public class Solution {
    public bool ContainsDuplicate(int[] nums) {
        HashSet<int> set = new HashSet<int>();
        int Length = nums.Length;
        if(Length <=1) return false;

            for(int i=0;  i<Length; i++)
            {
              if(set.Contains(nums[i]))
              {
                return true;   
              }
              else
              {        
                set.Add(nums[i]);
              }
            }

            return false;
    }
}
// Day 12 Class & Object
// 1603. Design Parking System
/*
Solution: Simulation
Time complexity: O(1) per addCar call
Space complexity: O(1)
*/
public class ParkingSystem {

    int[] count;
    public ParkingSystem(int big, int medium, int small) {
        count = new int[]{big, medium, small};
    }
    
    public bool AddCar(int carType) {
        return count[carType - 1]-- > 0;
    }
}

/**
 * Your ParkingSystem object will be instantiated and called as such:
 * ParkingSystem obj = new ParkingSystem(big, medium, small);
 * bool param_1 = obj.AddCar(carType);
 */

// 303. Range Sum Query - Immutable
/*
Solution: Prefix sum
sums[i] = nums[0] + nums[1] + … + nums[i]
sumRange(i, j) = sums[j] – sums[i – 1]

Time complexity: pre-compute: O(n), query: O(1)
Space complexity: O(n)*/
public class NumArray {

    int[] sums_; 
    public NumArray(int[] nums) {
        if (nums.Length == 0) return;
        sums_ = nums;
        for (int i = 1; i < nums.Length;++i)
          sums_[i] += sums_[i - 1];
    }
    
    public int SumRange(int left, int right) {
        if (left == 0) return sums_[right];
        return sums_[right] - sums_[left-1];
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * int param_1 = obj.SumRange(left,right);
 */