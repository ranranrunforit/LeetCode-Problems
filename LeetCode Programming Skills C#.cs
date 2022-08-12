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
public class Solution {
    public string FreqAlphabets(string s) {
        StringBuilder sb = new StringBuilder("");//used stringbuiled append b/c it's optimized
        int v = 0;
        int i = s.Length - 1;
        
        while (i >= 0) { //starts from last character, goes till first character
            if (s[i] == '//') {
                v = int.Parse(s.Substring(i-2, 2)) - 1;//using ascii, add 'a' to start from the alphabet, subtract '0' b/c currently the digits in the String s are chars
                i = i - 2;  //have skip 2 characters , b/c we already checked it in the above line
            } else {
                v = s[i] - '0' - 1;
            }
            sb.Append((char)('a' + v));
            i--;
        }
        //convert StringBuilder obj to string, reverses the whole string
        return new string(sb.ToString().Reverse().ToArray());
    }
}
// Regex
public class Solution {
    public string FreqAlphabets(string s) {
        return s.Replace("10//", "j")
            .Replace("11//", "k")
            .Replace("12//", "l")
            .Replace("13//", "m")
            .Replace("14//", "n")
            .Replace("15//", "o")
            .Replace("16//", "p")
            .Replace("17//", "q")
            .Replace("18//", "r")
            .Replace("19//", "s")
            .Replace("20//", "t")
            .Replace("21//", "u")
            .Replace("22//", "v")
            .Replace("23//", "w")
            .Replace("24//", "x")
            .Replace("25//", "y")
            .Replace("26//", "z")
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
/*Solution: Recursion
maxDepth(root) = max(maxDepth(root.left), maxDepth(root.right)) + 1

Time complexity: O(n)
Space complexity: O(n)*/
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

/* Iterative */
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


// Programming Skills II
// Day 1
// 896. Monotonic Array
public class Solution {
    public bool IsMonotonic(int[] nums) {
        bool inc = true;
        bool dec = true;

        for (int i = 1; i < nums.Length; ++i) {
          inc &= nums[i] >= nums[i - 1];
          dec &= nums[i] <= nums[i - 1];
        }

        return inc || dec;
    }
}

public class Solution {
    public bool IsMonotonic(int[] nums) {
        bool inc = true, dec = true;
        for (int i = 1; i < nums.Length; ++i) {
            inc &= nums[i - 1] <= nums[i];
            dec &= nums[i - 1] >= nums[i];
        }
        return inc || dec;
    }
}

// 28. Implement strStr()
public class Solution {
    public int StrStr(string haystack, string needle) {
        int l1 = haystack.Length;
        int l2 = needle.Length;
    for (int i = 0; i <= l1 - l2; ++i) {
      int j = 0;
      while (j < l2 && haystack[i + j] == needle[j]) ++j;
      if (j == l2) return i;
    }
    return -1;
    }
}

// Day 2
// 110. Balanced Binary Tree
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
    private bool balanced;
    public bool IsBalanced(TreeNode root) {
       this.balanced = true;
    height(root);
    return this.balanced;
  }
  
  private int height(TreeNode root) {
       if (root == null || this.balanced != true) return -1;
    int l = height(root.left);
    int r = height(root.right);
    if (Math.Abs(l - r) > 1) {
      this.balanced = false;
      return -1;
    }
    return Math.Max(l, r) + 1;
  }
}

// 459. Repeated Substring Pattern
/*
The time complexity here actually depends on how you compare string by using - contains(). 
In worst case, it could be O(m*n) where m and n are length of 2 strings respectively, 
in best case it's O(m+n) if you use KMP. 
To be honest, I have implemented KMP 3 or 4 times myself, 
but I am still not able to memorize details of it. :)
*/
public class Solution {
    public bool RepeatedSubstringPattern(string s) {
        String str = s + s;
        return str.Substring(1, str.Length - 2).Contains(s);
    }
}
/*
The length of the repeating substring must be a divisor of the length of the input string
Search for all possible divisor of str.length, starting for length/2
If i is a divisor of length, repeat the substring from 0 to i the number of times i is contained in s.length
If the repeated substring is equals to the input str return true
*/
public class Solution {
    public bool RepeatedSubstringPattern(string s) {
        int l = s.Length;
	for(int i=l/2;i>=1;i--) {
		if(l%i==0) {
			int m = l/i;
			String subS = s.Substring(0,i);
			StringBuilder sb = new StringBuilder();
			for(int j=0;j<m;j++) {
				sb.Append(subS);
			}
			if(sb.ToString().Equals(s)) return true;
		}
	}
	return false;
    }
}
// Day 3
// 150. Evaluate Reverse Polish Notation
public class Solution {
    public int EvalRPN(string[] tokens) {
        Stack<int> stack = new Stack<int>();
  
    for (int i = 0; i < tokens.Length; i++) {
        switch (tokens[i]) {
        case "+":
            stack.Push(stack.Pop() + stack.Pop());
            break;
            
        case "-":
            stack.Push(-stack.Pop() + stack.Pop());
            break;
            
        case "*":
            stack.Push(stack.Pop() * stack.Pop());
            break;

        case "/":
            int n1 = stack.Pop(), n2 = stack.Pop();
            stack.Push(n2 / n1);
            break;
            
        default:
            stack.Push(Int32.Parse(tokens[i]));
            break;
        }
    }
    
    return stack.Pop();
    }
}
/*
The Reverse Polish Notation is a stack of operations, 
thus, I decided to use java.util.Stack to solve this problem. 
As you can see, I add every token as an integer in the stack, 
unless it's an operation. 
In that case, I pop two elements from the stack and then save the result back to it. 
After all operations are done through, 
the remaining element in the stack will be the result.
Any comments or improvements are welcome.
*/
public class Solution {
    public int EvalRPN(string[] tokens) {
        int a,b;
		Stack<int> S = new Stack<int>();
		foreach (String s in tokens) {
			if(s.Equals("+")) {
				S.Push(S.Pop()+S.Pop());
			}
			else if(s.Equals("/")) {
				b = S.Pop();
				a = S.Pop();
				S.Push(a / b);
			}
			else if(s.Equals("*")) {
				S.Push(S.Pop() * S.Pop());
			}
			else if(s.Equals("-")) {
				b = S.Pop();
				a = S.Pop();
				S.Push(a - b);
			}
			else {
				S.Push(Int32.Parse(s));
			}
		}	
		return S.Pop();
    }
}

// 66. Plus One
public class Solution {
    public int[] PlusOne(int[] digits) {
        int n = digits.Length;
        for(int i=n-1; i>=0; i--) {
            if(digits[i] < 9) {
                digits[i]++;
                return digits;
            }

            digits[i] = 0;
        }

        int[] newNumber = new int [n+1];
        newNumber[0] = 1;

        return newNumber;
    }
}
// Day 4
// 1367. Linked List in Binary Tree
/*
Solution 2: DP
Iterate the whole link, find the maximum matched length of prefix.
Iterate the whole tree, find the maximum matched length of prefix.
About this dp, @fukuzawa_yumi gave a link of reference:
https://en.wikipedia.org/wiki/Knuth–Morris–Pratt_algorithm

Time O(N + L)
Space O(L + H)
where N = tree size, H = tree height, L = list length.*/

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
    public bool IsSubPath(ListNode head, TreeNode root) {
        List<int> A = new List<int>(), dp = new List<int>();
        A.Add(head.val);
        dp.Add(0);
        int i = 0;
        head = head.next;
        while (head != null) {
            while (i > 0 && head.val != A[i])
                i = dp[i - 1];
            if (head.val == A[i]) ++i;
            A.Add(head.val);
            dp.Add(i);
            head = head.next;
        }
        return dfs(root, 0, A, dp);
    }

    private bool dfs(TreeNode root, int i, List<int> A, List<int> dp) {
        if (root == null) return false;
        while (i > 0 && root.val != A[i])
            i = dp[i - 1];
        if (root.val == A[i]) ++i;
        return i == dp.Count || dfs(root.left, i, A, dp) || dfs(root.right, i, A, dp);
    }
}
/*
Solution 1: Brute DFS
Time O(N * min(L,H))
Space O(H)
where N = tree size, H = tree height, L = list length.
*/

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
    public bool IsSubPath(ListNode head, TreeNode root) {
         if (head == null) return true;
        if (root == null) return false;
        return dfs(head, root) || IsSubPath(head, root.left) || IsSubPath(head, root.right);
    }

    private bool dfs(ListNode head, TreeNode root) {
        if (head == null) return true;
        if (root == null) return false;
        return head.val == root.val && (dfs(head.next, root.left) || dfs(head.next, root.right));
    }
}
// 43. Multiply Strings

// Day 5
// 67. Add Binary
// Computation from string usually can be simplified by using a carry as such.
public class Solution {
    public string AddBinary(string a, string b) {
        StringBuilder sb = new StringBuilder();
        int i = a.Length - 1, j = b.Length -1, carry = 0;
        while (i >= 0 || j >= 0) {
            int sum = carry;
            if (j >= 0) sum += b[j--] - '0';
            if (i >= 0) sum += a[i--] - '0';
            sb.Append(sum % 2);
            carry = sum / 2;
        }
        if (carry != 0) sb.Append(carry);
        return new string(sb.ToString().Reverse().ToArray());
    }
}
// 989. Add to Array-Form of Integer
/*Take K itself as a Carry
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
*/
//With one loop.
public class Solution {
    public IList<int> AddToArrayForm(int[] num, int k) {
        IList<int> res = new List<int>();
        for (int i = num.Length - 1; i >= 0 || k > 0; --i) {
            res.Insert(0, (i >= 0 ? num[i] + k : k) % 10);
            k = (i >= 0 ? num[i] + k : k) / 10;
        }
        return res;
    }
}

public class Solution {
    public IList<int> AddToArrayForm(int[] num, int k) {
        IList<int> res = new List<int>();
        for (int i = num.Length - 1; i >= 0; --i) {
            res.Insert(0, (num[i] + k) % 10);
            k = (num[i] + k) / 10;
        }
        while (k > 0) {
            res.Insert(0, k % 10);
            k /= 10;
        }
        return res;
    }
}
// Day 6
// 739. Daily Temperatures
//Array
public class Solution {
    public int[] DailyTemperatures(int[] temperatures) {
        int[] stack = new int[temperatures.Length];
        int top = -1;
        int[] ret = new int[temperatures.Length];
        for(int i = 0; i < temperatures.Length; i++) {
            while(top > -1 && temperatures[i] > temperatures[stack[top]]) {
                int idx = stack[top--];
                ret[idx] = i - idx;
            }
            stack[++top] = i;
        }
        return ret;
    }
}
/*
Solution: Stack
Use a stack to track indices of future warmer days. From top to bottom: recent to far away.

Time complexity: O(n)

Space complexity: O(n)
*/
public class Solution {
    public int[] DailyTemperatures(int[] temperatures) {
        Stack<int> stack = new Stack<int>();
        int[] ret = new int[temperatures.Length];
        for(int i = 0; i < temperatures.Length; i++) {
            while(stack.Count != 0 && temperatures[i] > temperatures[stack.Peek()]) {
                int idx = stack.Pop();
                ret[idx] = i - idx;
            }
            stack.Push(i);
        }
        return ret;
    }
}
// 58. Length of Last Word
public class Solution {
    public int LengthOfLastWord(string s) {
        int i = s.Length - 1;
        int l = 0;
        while (i >= 0 && s[i] == ' ') --i;
        while (i >= 0 && s[i] != ' ') {
          --i;
          ++l;
        }
        return l;
    }
}
// Day 7
// 48. Rotate Image
// Approach 1: Rotate Groups of Four Cells
public class Solution {
    public void Rotate(int[][] matrix) {
        int n = matrix.Length;
        for (int i = 0; i < (n + 1) / 2; i ++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1];
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i];
                matrix[j][n - 1 - i] = matrix[i][j];
                matrix[i][j] = temp;
            }
        }
    }
}
/*
Solution: 2 Passes
First pass: mirror around diagonal
Second pass: mirror around y axis

Time complexity: O(n^2)
Space complexity: O(1)
*/
public class Solution {
    public void Rotate(int[][] matrix) {
        int n = matrix.Length;
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
      }
    }
        
    for (int i = 0; i < n; ++i)
      Array.Reverse(matrix[i]);
    }
}
// 1886. Determine Whether Matrix Can Be Obtained By Rotation
/*
Solution: Simulation
Time complexity: O(n2)
Space complexity: O(1)
*/
public class Solution {
    public bool FindRotation(int[][] mat, int[][] target) {
        int n = mat.Length;
        int[][] rot(int[][] mat) {      
          for (int i = 0; i < n; ++i){
              for (int j = i; j < n; ++j){
                int temp = mat[i][j]; 
                mat[i][j] = mat[j][i];
                mat[j][i] = temp;
            }
          }

          for (int m = 0; m < n; m++){
              for (int k = 0; k < n / 2; ++k){
                int temp1 = mat[k][m]; 
                mat[k][m] = mat[n - k - 1][m];
                mat[n - k - 1][m] = temp1;
            }
          }
          return mat;
        };
        for (int i = 0; i < 4; ++i){ 
          if (isEqual(rot(mat) , target)) return true; 
        }   
        return false;
    }
    public bool isEqual(int[][] mat, int[][] target){
        bool isEqual = true;
        for(int i=0;i< mat.Length; i++){
            for(int j=0;j<mat.Length; j++){
                if(mat[i][j] != target[i][j]){
                    isEqual = false;
                    break;
                }
            }
        }
        
        return isEqual;
    }
    
}
/*
Nothing Fancy to be done in the question. Try to break down the question into steps and solve.
To rotate a matrix inplace :

Find Transpose of the matrix
Swap the row element from start with end.
After that, check for each rotation if it is equal to target, 
if it is then return true, else again rotate the matrix. 
We can rotate the matrix maximum 4 times.
*/
public class Solution {
    public bool FindRotation(int[][] mat, int[][] target) {
        for(int i=0;i<4;i++){
            mat = rotate(mat);
            if(isEqual(mat, target)){
                return true;
            }
        }
        return false;
    }
    
    public bool isEqual(int[][] mat, int[][] target){
        bool isEqual = true;
        for(int i=0;i< mat.Length; i++){
            for(int j=0;j<mat.Length; j++){
                if(mat[i][j] != target[i][j]){
                    isEqual = false;
                    break;
                }
            }
        }
        
        return isEqual;
    }
    
    public int[][] rotate(int[][] mat){
        // First find transpose, i.e swap across major diagonal
        for(int i=0;i<mat.Length;i++){
            for(int j=0;j<i;j++){
                    int temp = mat[i][j];
                    mat[i][j] = mat[j][i];
                    mat[j][i] = temp;
            }
        }
        // Second swap rows across middle
        for(int i=0;i<mat.Length; i++){
            for(int j=0;j<mat.Length/2;j++){
                int temp = mat[i][j];
                mat[i][j] = mat[i][mat.Length-1-j];
                mat[i][mat.Length-1-j] = temp;
            }
        }
        
        return mat;
    }
}


// Day 8
// 54. Spiral Matrix
/*
Solution: Simulation

Keep track of the current bounds (left, right, top, bottom).

Init: left = 0, right = n – 1, top = 0, bottom = m – 1

Each time we move in one direction and shrink the bounds and turn 90 degrees:
1. go right => –top
2. go down => –right
3. go left => ++bottom
4. go up => ++left
*/
public class Solution {
    public IList<int> SpiralOrder(int[][] matrix) {
        if (matrix.Length == 0) return new List<int>();
        int l = 0;
        int t = 0;
        int r = matrix[0].Length;
        int b = matrix.Length;
        int total = (r--) * (b--);      
        int d = 0;
        int x = 0;
        int y = 0;
        List<int> ans = new List<int>();
        while (ans.Count < total - 1) {      
          if (d == 0) {
            while (x < r) ans.Add(matrix[y][x++]);
            ++t;
          } else if (d == 1) {
            while (y < b) ans.Add(matrix[y++][x]);
            --r;
          } else if (d == 2) {
            while (x > l) ans.Add(matrix[y][x--]);
            --b;
          } else if (d == 3) {
            while (y > t) ans.Add(matrix[y--][x]);
            ++l;
          }
          d = (d + 1) % 4;
        }
        if (ans.Count != total) ans.Add(matrix[y][x]);
        return ans;
    }
}
// 973. K Closest Points to Origin

// Day 9
// 1630. Arithmetic Subarrays
/*
Solution: Brute Force
Sort the range of each query and check.

Time complexity: O(nlogn * m)
Space complexity: O(n)
*/
public class Solution {
    public IList<bool> CheckArithmeticSubarrays(int[] nums, int[] l, int[] r) {
        bool[] ans = new bool[l.Length]; Array.Fill(ans, true);
    for (int i = 0; i < l.Length; ++i) {
        int[] arr = new int[r[i] - l[i] + 1];
        Array.Copy(nums, l[i], arr, 0, r[i] - l[i] + 1);
      Array.Sort(arr);
      int d = arr[1] - arr[0];
      for (int j = 2; j < arr.Length && ans[i]; ++j)
        ans[i] = ans[i] && (arr[j] - arr[j - 1] == d);
    }
    return ans.ToList();
    }
}

//linq
public class Solution {
    public IList<bool> CheckArithmeticSubarrays(int[] nums, int[] l, int[] r) =>
    l.Zip(r, (f, s) => nums[f..(s + 1)].OrderBy(n => n).ToArray())
     .Select(s => s.Zip(s.Skip(1), (c, n) => n - c).Distinct().Count() == 1)
     .ToArray();
}
// 429. N-ary Tree Level Order Traversal
/*Solution : Recursion
Time complexity: O(n)
Space complexity: O(n)*/
/*
// Definition for a Node.
public class Node {
    public int val;
    public IList<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, IList<Node> _children) {
        val = _val;
        children = _children;
    }
}
*/

public class Solution {
    public IList<IList<int>> LevelOrder(Node root) {
        IList<IList<int>> ans = new List<IList<int>>();
    preorder(root, 0, ans);
    return ans;
  }
private void preorder(Node root, int d, IList<IList<int>> ans) {
    if (root == null) return;
    while (ans.Count <= d) ans.Add(new List<int>());
    ans[d].Add(root.val);
    foreach (var child in root.children)
      preorder(child, d + 1, ans);
  
    }
}
// Solution : Iterative
/*
// Definition for a Node.
public class Node {
    public int val;
    public IList<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, IList<Node> _children) {
        val = _val;
        children = _children;
    }
}
*/

public class Solution {
    public IList<IList<int>> LevelOrder(Node root) {
        if (root == null) return new List<IList<int>>();
    IList<IList<int>> ans = new List<IList<int>>();    
    Queue<Node> q = new Queue<Node>();
    q.Enqueue(root);
    int depth = 0;
    while (q.Count != 0) {
      //int size = q.Count;
      ans.Add(new List<int>());
      for (int size = q.Count;size > 0;size--) {
        Node n = q.Peek(); q.Dequeue();
        ans[depth].Add(n.val);
        foreach (var child in n.children)
          q.Enqueue(child);
      }
      ++depth;
    }
    return ans;
    }
}
// Day 10
// 503. Next Greater Element II
/* Loop Twice
Explanation
Loop once, we can get the Next Greater Number of a normal array.
Loop twice, we can get the Next Greater Number of a circular array


Complexity
Time O(N) for one pass
Spce O(N) in worst case
*/
public class Solution {
    public int[] NextGreaterElements(int[] nums) {
        int n = nums.Length; int[] res = new int[n];
        Array.Fill(res, -1);
        Stack<int> stack = new Stack<int>();
        for (int i = 0; i < n * 2; i++) {
            while (stack.Count != 0 && nums[stack.Peek()] < nums[i % n])
                res[stack.Pop()] = nums[i % n];
            stack.Push(i % n);
        }
        return res;
    }
}
// 556. Next Greater Element III
// Next Permutation 
public class Solution {
    public int NextGreaterElement(int n) {
         int res = -1;
        char[] s = n.ToString().ToCharArray();
        int len = s.Count();
        for (int j = len - 2; j >= 0; j--) {
            for (int i = len - 1; i > j; i--) {
                if (s[i] > s[j]) {
                    Swap(s, i, j);
                    int k = len;
                    while (++j < --k) Swap(s, j, k);
                    return int.TryParse(new string(s), out res) ? res : -1;
                }
            }
        }
        
        return res;
    }
    
    private void Swap(char[] s, int a, int b) {
        if (a == b) return;
        char t = s[a];
        s[a] = s[b];
        s[b] = t;
    }
    
}
// Next Permutation 
public class Solution {
    public int NextGreaterElement(int n) {
         char[] arr = n.ToString().ToArray();
        int j = arr.Length - 2;
        for(; j >= 0; j--)
        {
            if(arr[j] < arr[j + 1])
                break;
        }
        
        if(j == -1)
            return -1;
                
        for(int i = arr.Length - 1; i > j; i--)
        {
            if(arr[i] > arr[j])
            {
                char temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                break;
            }
        }
        
        Reverse(arr, j + 1, arr.Length - 1);
        long res = Int64.Parse(new string(arr));
        return res <= Int32.MaxValue? Convert.ToInt32(res): -1;
    }
    
    private void Reverse(char[] arr, int i, int j)
    {
        while(i < j)
        {
            char temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
        }
    }
}
//Next Permutation 
public class Solution {
    public int NextGreaterElement(int n) {
        int[] arr = n.ToString().Select(x=>(int)(x-'0')).ToArray();
	int pivot = -1;
	int len = arr.Length;
	for (int i = len-1; i > 0; i--)
	{
		if (arr[i - 1] < arr[i])
		{
			pivot = i - 1;
			break;}
	}
	
	// If pivot is -1 it's the largest permutation, so return -1. in case of circular next permutation we reverse the whole array
	if(pivot == -1)  return -1;
	else
	{
		int index = -1;
		for (int i = len - 1; i >= pivot; i--)
		{
			if (arr[i] > arr[pivot])
			{
				index = i;
				break;
			}
		}
		Swap(arr, pivot, index);
		Array.Reverse(arr, pivot+1, len-pivot-1);		
	}
	long result = 0;
	foreach (var item in arr)
	{
		result = 10 * result + item;
	}
	
	// return -1 if the next permutation is greater than int32.MaxValue
	return result>Int32.MaxValue ? -1 : (int) result;

}

private void Swap(int[] arr, int i , int j)
{
	int temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
}

}
// Day 11
// 1376. Time Needed to Inform All Employees

// 49. Group Anagrams

// Day 12
// 438. Find All Anagrams in a String

// 713. Subarray Product Less Than K

// Day 13
// 304. Range Sum Query 2D - Immutable

// 910. Smallest Range II
/*
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
*/
public class Solution {
    public int SmallestRangeII(int[] nums, int k) {
        Array.Sort(nums);
        int n = nums.Length, mx = nums[n - 1], mn = nums[0], res = mx - mn;
        for (int i = 0; i < n - 1; ++i) {
            mx = Math.Max(mx, nums[i] + 2 * k);
            mn = Math.Min(nums[i + 1], nums[0] + 2 * k);
            res = Math.Min(res, mx - mn);
        }
        return res;
    }
}
// Day 14
// 143. Reorder List

// 138. Copy List with Random Pointer
/*Approach 1
Time: O(n)
Space: O(n)
*/
/*
// Definition for a Node.
public class Node {
    public int val;
    public Node next;
    public Node random;
    
    public Node(int _val) {
        val = _val;
        next = null;
        random = null;
    }
}
*/

public class Solution {
    public Node CopyRandomList(Node head) {
        if(head == null)
            return null;
        
        Dictionary<Node, Node> dic = new Dictionary<Node, Node>();
        
        // Deep copy nodes for values
        Node curr = head;
        while(curr != null)
        {
            dic.Add(curr, new Node(curr.val, null, null));
            curr = curr.next;
        }
        
        // Deep copy nodes for pointers
        curr = head;
        while(curr != null)
        {
            // key of dictionary can't be null
            dic[curr].next = curr.next == null? null : dic[curr.next];
            dic[curr].random = curr.random == null? null : dic[curr.random];
            curr = curr.next;
        }
        
        return dic[head];
    }
}
/*
Approach 2
Time: O(n): 3 passes
Space: O(1)
*/
/*
// Definition for a Node.
public class Node {
    public int val;
    public Node next;
    public Node random;
    
    public Node(int _val) {
        val = _val;
        next = null;
        random = null;
    }
}
*/

public class Solution {
    public Node CopyRandomList(Node head) {
        if(head == null)
            return null;
        
        // linked each node to its copy. 1 -> 1' -> 2 -> 2' -> 3 -> 3'
        Node curr = head;
        while(curr != null)
        {
            Node next = curr.next;          
            curr.next = new Node(curr.val);
            curr.next.next = next;
            
            curr = next;
        }
        
        // set random pointers for the copy nodes
        curr = head;
        while(curr != null)
        {
            if(curr.random != null)
                curr.next.random = curr.random.next;
            curr = curr.next.next;
        }
        
        // seperate copy linkedList from the original linkedList
        Node dummyHead = head.next;
        curr = head;
        Node curr2 = dummyHead;
        while(curr2.next != null)
        {    
            curr.next = curr.next.next;
            curr = curr.next;
            
            curr2.next = curr2.next.next;
            curr2 = curr2.next;
        }
        
        // handle the last original node
        curr.next = null;
        
        return dummyHead;
    }
}

/*
// Definition for a Node.
public class Node {
    public int val;
    public Node next;
    public Node random;
    
    public Node(int _val) {
        val = _val;
        next = null;
        random = null;
    }
}
*/

public class Solution {
    public Node CopyRandomList(Node head) {
        if (head == null)
        {
            return head;
        }
        
        Node h = head;
        Dictionary<Node, Node> dict = new Dictionary<Node, Node>();
        
        while (h != null)
        {
            dict.Add(h, new Node(h.val));
            
            h = h.next;
        }
        
        foreach (var n in dict.Keys)
        {
            dict[n].next = n.next == null ? null : dict[n.next];
            dict[n].random = n.random == null ? null : dict[n.random];
        }
        
        return dict[head];
    }
}
// Day 15
// 2. Add Two Numbers

// 445. Add Two Numbers II
/*
Solution: Simulation
Using a stack to “reverse” the list. Simulate the addition digit by digit.

Time complexity: O(l1 + l2)
Space complexity: O(l1 + l2)
*/
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
    public ListNode AddTwoNumbers(ListNode l1, ListNode l2) {
        Stack<int> s1 = new Stack<int>();
    Stack<int> s2 = new Stack<int>();
    while (l1 != null) {
      s1.Push(l1.val);
      l1 = l1.next;
    }    
    while (l2 != null) {
      s2.Push(l2.val);
      l2 = l2.next;
    }    
    ListNode head = null;
    int sum = 0;
    while (s1.Count != 0 || s2.Count != 0 || sum > 0) {
      sum += s1.Count == 0 ? 0 : s1.Peek();
      sum += s2.Count == 0 ? 0 : s2.Peek();
      if (s1.Count != 0 ) s1.Pop();
      if (s2.Count !=  0 ) s2.Pop();            
      ListNode n = new ListNode(sum % 10);
      sum /= 10;
      n.next = head;
      head = n;      
    }    
    return head;      
    }
}

// Day 16
// 61. Rotate List
/*
Solution: Find the prev of the new head

Step 1: Get the tail node T while counting the length of the list.
Step 2: k %= l, k can be greater than l, rotate k % l times has the same effect.
Step 3: Find the previous node P of the new head N by moving (l – k – 1) steps from head
Step 4: set P.next to null, T.next to head and return N

Time complexity: O(n) n is the length of the list
Space complexity: O(1)
*/
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
    public ListNode RotateRight(ListNode head, int k) {
        if (head == null) return null;
        int l = 1;
        ListNode tail = head;
        while (tail.next != null) {
          tail = tail.next;
          ++l;
        }
        k %= l;
        if (k == 0) return head;

        ListNode prev = head;
        for (int i = 0; i < l - k - 1; ++i) {
          prev = prev.next;
        }

        ListNode new_head = prev.next;
        prev.next = null;
        tail.next = head;
        return new_head;
    }
}

// 173. Binary Search Tree Iterator

// Day 17
// 1845. Seat Reservation Manager
/*
Solution: SortedSet
C# heap equivalent is SortedSet.
Time complexity: O(nlogn)
Space complexity: O(n)*/
public class SeatManager {

    public SeatManager(int n) {
        for(int i = 1; i <= n; i++)
            set.Add(i);
    }
    
    public int Reserve() {
        //var min = set.Min; this works too!
        var min = set.First();
        set.Remove(min);
        return min;
    }
    
    public void Unreserve(int seatNumber) {
         set.Add(seatNumber);
    }
    private SortedSet<int> set = new SortedSet<int>();
}

/**
 * Your SeatManager object will be instantiated and called as such:
 * SeatManager obj = new SeatManager(n);
 * int param_1 = obj.Reserve();
 * obj.Unreserve(seatNumber);
 */
//PriorityQueue
 public class SeatManager {

    public SeatManager(int n) {
        seats = new PriorityQueue<int,int>(Comparer<int>.Create((x, y) => x - y));
        for (int i = 1; i <= n; i++) {
            seats.Enqueue(i , i);
        }
    }
    
    public int Reserve() {
        return seats.Dequeue();
    }
    
    public void Unreserve(int seatNumber) {
         seats.Enqueue(seatNumber, seatNumber); 
    }
    private PriorityQueue<int,int> seats;
}

public class SeatManager {

    public SeatManager(int n) {
        for (int i = 1; i <= n; ++i)
            s_.Add(i);
    }
    
    public int Reserve() {
        int seat = s_.Min();
        s_.Remove(seat);    
        return seat;
    }
    
    public void Unreserve(int seatNumber) {
         s_.Add(seatNumber);  
    }
    private List<int> s_ = new List<int>(); // TLE List is too slow
}

/**
 * Your SeatManager object will be instantiated and called as such:
 * SeatManager obj = new SeatManager(n);
 * int param_1 = obj.Reserve();
 * obj.Unreserve(seatNumber);
 */
/**
 * Your SeatManager object will be instantiated and called as such:
 * SeatManager obj = new SeatManager(n);
 * int param_1 = obj.Reserve();
 * obj.Unreserve(seatNumber);
 */
// 860. Lemonade Change
/*
Intuition:
When the customer gives us $20, we have two options:

To give three $5 in return
To give one $5 and one $10.
On insight is that the second option (if possible) is always better than the first one.
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
*/
public class Solution {
    public bool LemonadeChange(int[] bills) {
        int five = 0, ten = 0;
        foreach (int i in bills) {
            if (i == 5) five++;
            else if (i == 10) {five--; ten++;}
            else if (ten > 0) {ten--; five--;}
            else five -= 3;
            if (five < 0) return false;
        }
        return true;
    }
}
// Day 18
// 155. Min Stack

// 341. Flatten Nested List Iterator

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool IsInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     int GetInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     IList<NestedInteger> GetList();
 * }
 */
public class NestedIterator {
    //DFS
    private Queue<int> q = new Queue<int>();
    public NestedIterator(IList<NestedInteger> nestedList) {
        foreach (var item in nestedList)
            DFS(item);
    }

    public bool HasNext() {
        return q.Count != 0;
    }

    public int Next() {
        return q.Count == 0 ? 0 : q.Dequeue();
    }
    
    private void DFS(NestedInteger cur)
    {
        if (cur.IsInteger())
            q.Enqueue(cur.GetInteger());
        else
            foreach (var item in cur.GetList())
                DFS(item);
    }
    
}

/**
 * Your NestedIterator will be called like this:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.HasNext()) v[f()] = i.Next();
 */

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool IsInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     int GetInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     IList<NestedInteger> GetList();
 * }
 */
public class NestedIterator {
    // Recursion
    List<int> list = new List<int>();
    int count = -1;
    public NestedIterator(IList<NestedInteger> nestedList) {
        addToList(nestedList);
    }

    public bool HasNext() {
        return count < list.Count -1;
    }

    public int Next() {
        count++;
        return list[count];  
    }
    
    public void addToList(IList<NestedInteger> nestedList)
    {
         foreach(NestedInteger i in nestedList)
        {
            if(i.IsInteger())
            {
                list.Add(i.GetInteger());
            }
            else
            {
               addToList(i.GetList()); 
            }
        }
    }
    
   
}

/**
 * Your NestedIterator will be called like this:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.HasNext()) v[f()] = i.Next();
 */

// Day 19
// 1797. Design Authentication Manager
/*
Solution: Hashtable
Use a hashtable to store the token and its expiration time.

Time complexity: at most O(n) per operation
Space complexity: O(n)
*/
public class AuthenticationManager {

    public AuthenticationManager(int timeToLive) {
        ttl_ = timeToLive;
    }
    
    public void Generate(string tokenId, int currentTime) {
        clear(currentTime);
        tokens_[tokenId] = currentTime + ttl_;
    }
    
    public void Renew(string tokenId, int currentTime) {
        clear(currentTime);
        if (tokens_.ContainsKey(tokenId)) tokens_[tokenId] = currentTime + ttl_; 
        else return; 
    }
    
    public int CountUnexpiredTokens(int currentTime) {
        clear(currentTime);
        return tokens_.Count;
    }
    
    private void clear(int currentTime) {
    List<String> ids = new List<String>();
    foreach (KeyValuePair<String, int> item in tokens_)
      if (item.Value <= currentTime) ids.Add(item.Key);
    foreach (String id in ids)
      tokens_.Remove(id);
  }
    
    Dictionary<String, int> tokens_ = new Dictionary<String, int>();
    int ttl_;
}

/**
 * Your AuthenticationManager object will be instantiated and called as such:
 * AuthenticationManager obj = new AuthenticationManager(timeToLive);
 * obj.Generate(tokenId,currentTime);
 * obj.Renew(tokenId,currentTime);
 * int param_3 = obj.CountUnexpiredTokens(currentTime);
 */
// 707. Design Linked List

// Day 20
// 380. Insert Delete GetRandom O(1)
/*
Idea: Hashtable + array
Time complexity: O(1)
*/
public class RandomizedSet {
    /** Initialize your data structure here. */
    public RandomizedSet() {
        
    }
    /** Inserts a value to the set. 
    Returns true if the set did not already contain the specified element. */
    public bool Insert(int val) {
        if(m_.ContainsKey(val)) return false;
        
        m_[val] = vals_.Count;
        vals_.Add(val);
        return true;
        
        
    }
    /** Removes a value from the set. 
    Returns true if the set contained the specified element. */
    public bool Remove(int val) {
        if(!m_.ContainsKey(val)) return false;
        int index = m_[val];
        m_[vals_[vals_.Count - 1]] = index;
        m_.Remove(val);
        int temp = vals_[vals_.Count - 1];
        vals_[vals_.Count - 1] = vals_[index];
        vals_[index] = temp; 
        vals_.RemoveAt(vals_.Count-1);
        return true;
    }
    /** Get a random element from the set. */
    public int GetRandom() {
        Random r = new Random();
        int index = r.Next(0,vals_.Count);
        return vals_[index];
    }
    
    
    // val -> index in the array
    private Dictionary<int, int> m_ = new Dictionary<int, int>();
    private List<int> vals_ = new List<int>();
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * bool param_1 = obj.Insert(val);
 * bool param_2 = obj.Remove(val);
 * int param_3 = obj.GetRandom();
 */
// 622. Design Circular Queue
/*
Solution: Simulate with an array
We need a fixed length array, 
and the head location as well as the size of the current queue.

We can use q[head] to access the front, 
and q[(head + size – 1) % k] to access the rear.

Time complexity: O(1) for all the operations.
Space complexity: O(k)
*/
public class MyCircularQueue {
    private int[] q;
  private int head;
  private int size;

    public MyCircularQueue(int k) {
        this.q = new int[k];
    this.head = 0;
    this.size = 0;
    }
    
    public bool EnQueue(int value) {
        if (this.IsFull()) return false;    
    this.q[(this.head + this.size) % this.q.Length] = value;
    ++this.size;
    return true;
    }
    
    public bool DeQueue() {
        if (this.IsEmpty()) return false;
    this.head = (this.head + 1) % this.q.Length;
    --this.size;
    return true;
    }
    
    public int Front() {
        return this.IsEmpty() ? -1 : this.q[this.head];
    }
    
    public int Rear() {
        return this.IsEmpty() ? -1 : this.q[(this.head + this.size - 1) % this.q.Length];
    }
    
    public bool IsEmpty() {
         return this.size == 0;
    }
    
    public bool IsFull() {
        return this.size == this.q.Length;
    }
}

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * bool param_1 = obj.EnQueue(value);
 * bool param_2 = obj.DeQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * bool param_5 = obj.IsEmpty();
 * bool param_6 = obj.IsFull();
 */

// 729. My Calendar I
// SortedList + BinarySearch Solution
public class MyCalendar {
 SortedList<int, int> calendar;
    public MyCalendar() {
        calendar = new SortedList<int, int>();
    }
    
    public bool Book(int start, int end) {
        int l = 0, r = calendar.Count - 1, m = 0;
        while (l <= r)
        {
            m = l + (r - l) / 2;
            var _start = calendar.Keys[m];
            var _end = calendar.Values[m];
            if ((start >= _start && start < _end) ||
                ( start < _start && end > _start))
                return false;
            if (start > _start)
                l = m + 1;
            else
                r = m - 1;                
        }
        
        calendar.Add(start, end);
        return true;
    }
}

/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar obj = new MyCalendar();
 * bool param_1 = obj.Book(start,end);
 */ 

/*
Solution : Brute Force: O(n^2)
*/
 public class MyCalendar {
List<int[]> calendar;
    public MyCalendar() {
        calendar = new List<int[]>();
    }
    
    public bool Book(int start, int end) {
        foreach (int[] iv in calendar) {
            if (iv[0] < end && start < iv[1]) return false;
        }
        calendar.Add(new int[]{start, end});
        return true;
    }
}

/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar obj = new MyCalendar();
 * bool param_1 = obj.Book(start,end);
 */