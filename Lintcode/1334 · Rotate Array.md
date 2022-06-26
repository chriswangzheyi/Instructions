# 1334 · Rotate Array

## 题目

Given an array, rotate the array to the right by k steps, where k is non-negative.

Example 1:

	Input: [1,2,3,4,5,6,7], k = 3
	Output: [5,6,7,1,2,3,4]
	Explanation:
	rotate 1 steps to the right: [7,1,2,3,4,5,6]
	rotate 2 steps to the right: [6,7,1,2,3,4,5]
	rotate 3 steps to the right: [5,6,7,1,2,3,4]
Example 2:

	Input: [-1,-100,3,99], k = 2
	Output: [3,99,-1,-100]
	Explanation: 
	rotate 1 steps to the right: [99,-1,-100,3]
	rotate 2 steps to the right: [3,99,-1,-100]


## 代码

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param nums: an array
	    @param k: an integer
	    @return: rotate the array to the right by k steps
	    """
	    def rotate(self, nums: List[int], k: int) -> List[int]:
	        # Write your code here
	        k %= len(nums)
	        return nums[-k:] + nums[:-k]
	        
## 解释
举例：

输入: [1,2,3,4,5,6,7], k = 3

* k= 3 （ 3 % 7 取模等于3 ）
* nums[-k:] =     [5, 6, 7]
* nums[:-k]. =    [1, 2, 3, 4]

