# 9 · Fizz Buzz

## 题目

Given number n. Print number from 1 to n. According to following rules:

* when number is divided by 3, print "fizz".
* when number is divided by 5, print "buzz".
* when number is divided by both 3 and 5, print "fizz buzz".
* when number can't be divided by either 3 or 5, print the number itself.

Example 1:

Input:

	n = 15
Output:

	[
	  "1", "2", "fizz",
	  "4", "buzz", "fizz",
	  "7", "8", "fizz",
	  "buzz", "11", "fizz",
	  "13", "14", "fizz buzz"
	]

## 代码

	from typing import (
	    List,
	)
	
	class Solution:
	    """
	    @param n: An integer
	    @return: A list of strings.
	    """
	    def fizz_buzz(self, n: int) -> List[str]:
	        # write your code here
	        answer = []
	        for i in range(1, n+1):
	            if i % 15 == 0:
	                answer.append("fizz buzz")
	            elif i % 5 == 0:
	                answer.append("buzz")
	            elif i % 3 == 0:
	                answer.append("fizz")
	            else:
	                answer.append(str(i))
	        return answer

## 容易错的点：

for 循环，开始必须为1， 结束需要是长度加1（因为是左闭右开）