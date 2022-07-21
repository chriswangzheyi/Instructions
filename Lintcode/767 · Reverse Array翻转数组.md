# 767 · Reverse Array

## 题目

Reverse the given array nums inplace.

Example 1:

	Input : nums = [1,2,5]
	Output : [5,2,1]
	
## 代码 （双指针）

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param nums: a integer array
	    @return: nothing
	    """
	    def reverse_array(self, nums: List[int]):
	        # write your code here
	        left, right = 0, len(nums) - 1
	        
	        while left <= right:
	            nums[left], nums[right] = nums[right], nums[left]
	            left += 1
	            right -= 1
	        return nums
