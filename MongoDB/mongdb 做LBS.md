**基于MongoDB做LBS**



**1. 版本:**   

 MongoDB 4.2


**2. GeoJSON格式**

    { "type": "Point", "coordinates": [lon(经度),lat(纬度)]}


**3.测试数据**

往数据库中添加：

	db.geo.insert({"address" : "南京 禄口国际机场","loc" : { "type": "Point", "coordinates": [118.783799,31.979234]}})
	db.geo.insert({"address" : "南京 浦口公园","loc" : { "type": "Point", "coordinates": [118.639523,32.070078]}})
	db.geo.insert({"address" : "南京 火车站","loc" : { "type": "Point", "coordinates": [118.803032,32.09248]}})
	db.geo.insert({"address" : "南京 新街口","loc" : { "type": "Point", "coordinates": [118.790611,32.047616]}})
	db.geo.insert({"address" : "南京 张府园","loc" : { "type": "Point", "coordinates": [118.790427,32.03722]}})
	db.geo.insert({"address" : "南京 三山街","loc" : { "type": "Point", "coordinates": [118.788135,32.029064]}})
	db.geo.insert({"address" : "南京 中华门","loc" : { "type": "Point", "coordinates": [118.781161,32.013023]}})
	db.geo.insert({"address" : "南京 安德门","loc" : { "type": "Point", "coordinates": [118.768964,31.99646]}})

其中：geo为collection的名称



**添加索引**

    db.geo.createIndex( { loc : "2dsphere" } )


**4.验证**


检索规定半径以内数据（单位为米）
	
	 db.geo.find({loc:{$near: {$geometry: {type: "Point" ,coordinates: [118.783799,31.979234]},$maxDistance: 5000}}})





**5.语法**

    
    {
       <location field>: {
     $near: {
       $geometry: {
      type: "Point" ,
      coordinates: [ <longitude> , <latitude> ]
       },
       $maxDistance: <distance in meters>,
       $minDistance: <distance in meters>
       }
      }
    }


> 参考资料：https://docs.mongodb.com/manual/reference/operator/query/near/