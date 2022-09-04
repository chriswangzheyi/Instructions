# Stream Demo

参考：https://howtodoinjava.com/java/stream/java-streams-by-examples/

## 创建steam

### Stream.of()

	Stream<Integer> stream = Stream.of(1,2,3,4,5,6,7,8,9);
	stream.forEach(p -> System.out.println(p));

### Stream.of(array)

	Stream<Integer> stream = Stream.of( new Integer[]{1,2,3,4,5,6,7,8,9} );
	stream.forEach(p -> System.out.println(p));

### List.stream()

	List<Integer> list = new ArrayList<Integer>();
	
	for(int i = 1; i< 10; i++){
	      list.add(i);
	}
	
	Stream<Integer> stream = list.stream();
	stream.forEach(p -> System.out.println(p));
        
### Stream.generate() or Stream.iterate()

        Stream stream = Stream.generate( () -> (new Random().nextInt(100)) );
        stream.forEach(p -> System.out.println(p));

### Stream of String chars or tokens

	IntStream stream = "12345_abcdefg".chars();
	stream.forEach(p -> System.out.println(p));
	
	//OR
	
	Stream<String> stream = Stream.of("A$B$C".split("\\$"));
	stream.forEach(p -> System.out.println(p));
	
## Stream Collectors

### Collect Stream elements to a List

        List<Integer> list = new ArrayList<Integer>();

        for(int i = 1; i< 10; i++){
            list.add(i);
        }

        Stream<Integer> stream = list.stream();

        List<Integer> evenNumbersList = stream.filter(i -> i%2 == 0)
                .collect(Collectors.toList());

        System.out.print(evenNumbersList);
        
### Collect Stream elements to an Array

	List<Integer> list = new ArrayList<Integer>();
	 
	for(int i = 1; i< 10; i++){
	      list.add(i);
	}
	
	Stream<Integer> stream = list.stream();
	Integer[] evenNumbersArr = stream.filter(i -> i%2 == 0).toArray(Integer[]::new);
	System.out.print(evenNumbersArr);   
	
## 计算 Stream Operations

### 准备

	List<String> memberNames = new ArrayList<>();
	memberNames.add("Amitabh");
	memberNames.add("Shekhar");
	memberNames.add("Aman");
	memberNames.add("Rahul");
	memberNames.add("Shahrukh");
	memberNames.add("Salman");
	memberNames.add("Yana");
	memberNames.add("Lokesh");
	
### Stream.filter() 

	memberNames.stream().filter((s) -> s.startsWith("A"))
                    .forEach(System.out::println);

### Stream.map()

        memberNames.stream().map(String::toUpperCase)
                .forEach(System.out::println);

### Stream.sorted()

        memberNames.stream().sorted()
                .forEach(System.out::println);
         
                
### Stream().flatMap                

        List<Integer> list1 = Arrays.asList(1,2,3);
        List<Integer> list2 = Arrays.asList(4,5,6);
        List<Integer> list3 = Arrays.asList(7,8,9);

        List<List<Integer>> listOfLists = Arrays.asList(list1, list2, list3);

        List<Integer> listOfAllIntegers = listOfLists.stream()
                .flatMap(x -> x.stream())
                .collect(Collectors.toList());

        System.out.println(listOfAllIntegers);

                
### Stream().distinct     
 
	 Collection<String> list = Arrays.asList("A", "B", "C", "D", "A", "B", "C");
	 
	// Get collection without duplicate i.e. distinct only
	List<String> distinctElements = list.stream()
	                        .distinct()
	                        .collect(Collectors.toList());
	 
	//Let's verify distinct elements
	System.out.println(distinctElements);               
                
## 退出 Terminal operations

### Stream.forEach()

	memberNames.forEach(System.out::println);
	
### Stream.collect()

	List<String> memNamesInUppercase = memberNames.stream().sorted()
	                            .map(String::toUpperCase)
	                            .collect(Collectors.toList());
	
	System.out.print(memNamesInUppercase)

### Stream.match()

返回 true 或者 false

	boolean matchedResult = memberNames.stream()
	        .anyMatch((s) -> s.startsWith("A"));
	 
	System.out.println(matchedResult);     //true
	 
	matchedResult = memberNames.stream()
	        .allMatch((s) -> s.startsWith("A"));
	 
	System.out.println(matchedResult);     //false
	 
	matchedResult = memberNames.stream()
	        .noneMatch((s) -> s.startsWith("A"));
	 
	System.out.println(matchedResult);     //false
	
### Stream.count()

	long totalMatched = memberNames.stream()
	    .filter((s) -> s.startsWith("A"))
	    .count();
	 
	System.out.println(totalMatched);     //2
	
### Stream.reduce()

	Optional<String> reduced = memberNames.stream()
	        .reduce((s1,s2) -> s1 + "#" + s2);
	 
	reduced.ifPresent(System.out::println);
	
输出结果：

	Amitabh#Shekhar#Aman#Rahul#Shahrukh#Salman#Yana#Lokesh

## 缩减 Short-circuit Operations

###  Stream.anyMatch()

	boolean matched = memberNames.stream()
	        .anyMatch((s) -> s.startsWith("A"));
	 
	System.out.println(matched);    //true
	
### Stream.findFirst()

	String firstMatchedName = memberNames.stream()
	            .filter((s) -> s.startsWith("L"))
	            .findFirst()
	                        .get();
	 
	System.out.println(firstMatchedName);    //Lokesh
	
## 并行流 Parallel Streams

        List<Integer> list = new ArrayList<Integer>();
        for(int i = 1; i< 10; i++){
            list.add(i);
        }

        //Here creating a parallel stream
        Stream<Integer> stream = list.parallelStream();

        Integer[] evenNumbersArr = stream.filter(i -> i%2 == 0).toArray(Integer[]::new);
        System.out.print(evenNumbersArr);

比传统的串行stream快

  