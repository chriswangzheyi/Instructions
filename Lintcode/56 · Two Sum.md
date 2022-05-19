# 56 · Two Sum

## 题目

Given an array of integers, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are zero-based.


样例
Example 1:

Input:

	numbers = [2,7,11,15]
	target = 9
	
Output:

	[0,1]
	
Explanation:

numbers[0] + numbers[1] = 9


Example 2:

Input:

	numbers = [15,2,7,11]
	target = 9
	
Output:

	[1,2]
	
Explanation:

numbers[1] + numbers[2] = 9

## 代码 

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param numbers: An array of Integer
	    @param target: target = numbers[index1] + numbers[index2]
	    @return: [index1, index2] (index1 < index2)
	    """
	    def two_sum(self, numbers: List[int], target: int) -> List[int]:
	        # write your code here
	        hash = {}
	
	        for i, num in enumerate(numbers):
	            if target - num in hash:
	                return [hash[target-num],i]
	            hash[num] = i
	            
	        return hash

## 代码2 (双指针解法)

 