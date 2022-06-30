# 75 · Find Peak Element

There is an integer array which has the following features:

The numbers in adjacent positions are different.
A[0] < A[1] && A[A.length - 2] > A[A.length - 1].
We define a position P is a peak if:

	A[P] > A[P-1] && A[P] > A[P+1]
Find a peak element in this array. Return the index of the peak.


Example 1:

Input:

	A = [1, 2, 1, 3, 4, 5, 7, 6]
Output:

	1
Explanation:

Returns the index of any peak element. 6 is also correct.
Example 2:

Input:

	A = [1,2,3,4,1]
Output:

	3
Explanation:

return the index of peek.

## 代码

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param a: An integers array.
	    @return: return any of peek positions.
	    """
	    def find_peak(self, a: List[int]) -> int:
	        # write your code here
	        start, end = 1, len(A) - 2
	        while start + 1 <  end:
	            mid = (start + end) // 2
	            if A[mid] < A[mid - 1]:
	                end = mid
	            elif A[mid] < A[mid + 1]:
	                start = mid
	            else:
	                return mid
	
	        if A[start] < A[end]:
	            return end
	        else:
	            return start

## 	解释

这个题 LintCode 和 LeetCode 的 find peak element 是有区别的。
数据上，LintCode 保证数据第一个数比第二个数小，倒数第一个数比到倒数第二个数小。
因此 start, end 的范围要取 1, len(A) - 2

二分法。
每次取中间元素，如果大于左右，则这就是peek。
否则取大的一边，两个都大，可以随便取一边。最终会找到peek。

正确性证明：

A[0] < A[1] && A[n-2] > A[n-1] 所以一定存在一个peek元素。
二分时，选择大的一边, 则留下的部分仍然满足1 的条件，即最两边的元素都小于相邻的元素。所以仍然必然存在peek。
二分至区间足够小，长度为3, 则中间元素就是peek。