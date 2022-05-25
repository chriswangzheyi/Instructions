# 82 · Single Number

## 题目

Given 2 * n + 1 numbers, every numbers occurs twice except one, find it.

Example 1:

Input:

	A = [1,1,2,2,3,4,4]
	
Output:

	3
	
Explanation:

Only 3 appears once

Example 2:

Input:

	A = [0,0,1]

Output:

	1

Explanation:

Only 1 appears oncears once


## 代码 （字典法）

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param a: An integer array
	    @return: An integer
	    """
	    def single_number(self, a: List[int]) -> int:
	        # write your code here
	        if not a:
	            return -1
	        
	        dic = {}
	        for i in a:
	            dic[i] = dic.get(i, 0) + 1 
	
	        for key, value in dic.items():
	            if value == 1:
	                return key

