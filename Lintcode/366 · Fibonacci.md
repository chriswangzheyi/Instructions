# 366 · Fibonacci

## 题目

Find the Nth number in Fibonacci sequence. (N starts at 0)

A Fibonacci sequence is defined as follow:

The first two numbers are 0 and 1.
The i th number is the sum of i-1 th number and i-2 th number.
The first ten numbers in Fibonacci sequence is:

0, 1, 1, 2, 3, 5, 8, 13, 21, 34 ...


	Example 1:
		Input:  1
		Output: 0
		
		Explanation: 
		return the first number in  Fibonacci sequence .
	
	Example 2:
		Input:  2
		Output: 1
		
		Explanation: 
		return the second number in  Fibonacci sequence .
		

## 代码1

	class Solution:
	    """
	    @param n: an integer
	    @return: an ineger f(n)
	    """
	    def fibonacci(self, n: int) -> int:
	        # write your code here
	        a = 0
	        b = 1
	        for i in range(n - 1):
	            a, b = b, a + b
	        return a
    
## 代码2 (递归)

	class Solution:
	    """
	    @param n: an integer
	    @return: an ineger f(n)
	    """
	    def fibonacci(self, n: int) -> int:
	        # write your code here
	        if n == 1 or n == 2:
	            return n - 1
	            
	        return self.fibonacci(n - 1)+ self.fibonacci(n-2)
	  
容易出现：Time Limit Exceeded