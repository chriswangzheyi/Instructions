**一致性Hash(加虚拟节点)的Java实现**



    import java.util.ArrayList;
    import java.util.HashMap;
    import java.util.LinkedList;
    import java.util.List;
    import java.util.Set;
    import java.util.SortedMap;
    import java.util.TreeMap;
    import java.util.UUID;
    
    public class ConsistentHashingWithVirtualNode {
    /**
     * 待添加入Hash环的服务器列表
     */
    private static String[] servers = { "192.168.0.0:111","192.168.0.1:111", "192.168.0.2:111"};

    /**
     * 真实结点列表,考虑到服务器上线、下线的场景，即添加、删除的场景会比较频繁，这里使用LinkedList会更好
     */
    private static List<String> realNodes = new LinkedList<String>();
    /**
     * 虚拟节点，key表示虚拟节点的hash值，value表示虚拟节点的名称
     */
    private static SortedMap<Integer, String> virtualNodes =
            new TreeMap<Integer, String>();

    /**
     * 虚拟节点的数目，这里写死，为了演示需要，一个真实结点对应5个虚拟节点
     */
    private static final int VIRTUAL_NODES = 1000;

    static
    {
        // 先把原始的服务器添加到真实结点列表中
        for (int i = 0; i < servers.length; i++)
            realNodes.add(servers[i]);

        // 再添加虚拟节点，遍历LinkedList使用foreach循环效率会比较高
        for (String str : realNodes)
        {
            for (int i = 0; i < VIRTUAL_NODES; i++)
            {
                String virtualNodeName = str + "&&VN" + String.valueOf(i);
                int hash = getHash(virtualNodeName);
                System.out.println("虚拟节点[" + virtualNodeName + "]被添加, hash值为" + hash);
                virtualNodes.put(hash, virtualNodeName);
            }
        }
        System.out.println();
    }

    /**
     * 使用FNV1_32_HASH算法计算服务器的Hash值,这里不使用重写hashCode的方法，最终效果没区别
     */
    private static int getHash(String str)
    {
        final int p = 16777619;
        int hash = (int)2166136261L;
        for (int i = 0; i < str.length(); i++)
            hash = (hash ^ str.charAt(i)) * p;
        hash += hash << 13;
        hash ^= hash >> 7;
        hash += hash << 3;
        hash ^= hash >> 17;
        hash += hash << 5;

        // 如果算出来的值为负数则取其绝对值
        if (hash < 0)
            hash = Math.abs(hash);
        return hash;
    }

    /**
     * 得到应当路由到的结点
     */
    private static String getServer(String node)
    {
        // 得到带路由的结点的Hash值
        int hash = getHash(node);
        // 得到大于该Hash值的所有Map
        SortedMap<Integer, String> subMap =
                virtualNodes.tailMap(hash);
        Integer i=null;
        String virtualNode = null;
        if(subMap==null||subMap.size()==0){
            i=virtualNodes.firstKey();
            virtualNode=virtualNodes.get(i);
        }else{
            i = subMap.firstKey();
            virtualNode= subMap.get(i);
        }
        // 第一个Key就是顺时针过去离node最近的那个结点

        // 返回对应的虚拟节点名称，这里字符串稍微截取一下
        return virtualNode.substring(0, virtualNode.indexOf("&&"));
    }

    public static void main(String[] args)
    {

        HashMap<String,Integer> map=new HashMap<String, Integer>();
        List<String> id=new ArrayList<String>();
        for(int i=0;i<1000;i++){
            id.add(UUID.randomUUID().toString().replace("-", ""));
            //id.add("adasfdsafdsgfdsagdsafdsafdsaf"+i);
        }
        for (int i = 0; i < id.size(); i++){
            String aString=getServer(id.get(i));
            Integer aInteger=map.get(aString);
            if(aInteger==null){
                map.put(aString,1);
            }else{
                map.put(aString, aInteger+1);
            }
        }
        Set<String> set= map.keySet();
        for(String a:set){
            System.out.println("节点【"+a+"】分配到元素个数为==>"+map.get(a));
        }
      }
    }




  