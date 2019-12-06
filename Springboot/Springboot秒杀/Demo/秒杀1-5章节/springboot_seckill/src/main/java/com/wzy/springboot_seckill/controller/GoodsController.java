package com.wzy.springboot_seckill.controller;


import com.wzy.springboot_seckill.entity.Evaluate;
import com.wzy.springboot_seckill.entity.Goods;
import com.wzy.springboot_seckill.service.GoodsService;
import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import javax.annotation.Resource;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
public class GoodsController {
    Logger logger = LoggerFactory.getLogger(GoodsController.class);

    @Resource
    private GoodsService goodsService;

    //在SpringBoot IOC容器初始化的时候，自动Configuration就被实例化
    @Resource
    private Configuration freemarkerConfig;

    @GetMapping("/goods") //http://localhost:8080/goods?gid=739
    @ResponseBody
    public ModelAndView showGoods(Long gid){
        logger.info("gid:" + gid);
        ModelAndView mav = new ModelAndView("/goodsview");
        Goods goods = goodsService.getGoods(gid);

        mav.addObject("goods", goods);
        mav.addObject("covers", goodsService.findCovers(gid));
        mav.addObject("details", goodsService.findDetails(gid));
        mav.addObject("params", goodsService.findParams(gid));
        return mav;
    }

    @GetMapping("/static/{gid}")  //http://localhost:8080/static/739
    @ResponseBody
    public String doStatic(@PathVariable("gid") Long gid) throws IOException, TemplateException {
        //获取模板对象
        Template template = freemarkerConfig.getTemplate("goodsview.ftl");
        Map param = new HashMap();
        param.put("goods", goodsService.getGoods(gid));
        param.put("covers", goodsService.findCovers(gid));
        param.put("details", goodsService.findDetails(gid));
        param.put("params", goodsService.findParams(gid));
        File targetFile = new File("E:/babytun/goods/" + gid + ".html");
        FileWriter out = new FileWriter(targetFile);
        template.process(param , out);
        out.close();
        return targetFile.getPath();
    }

    @GetMapping("/static_all")   //http://localhost:8080/static_all
    @ResponseBody
    public String doStatic() throws IOException, TemplateException {
        //获取模板对象
        Template template = freemarkerConfig.getTemplate("goodsview.ftl");
        List<Goods> allGoods = goodsService.findAllGoods();
        for (Goods g : allGoods) {
            Long gid = g.getGoodsId();
            Map param = new HashMap();
            param.put("goods", goodsService.getGoods(gid));
            param.put("covers", goodsService.findCovers(gid));
            param.put("details", goodsService.findDetails(gid));
            param.put("params", goodsService.findParams(gid));
            File targetFile = new File("E:/babytun/goods/" + gid + ".html");
            FileWriter out = new FileWriter(targetFile);
            template.process(param , out);
            out.close();
        }

        return "ok";
    }
    @GetMapping("/evaluate/{gid}")
    @ResponseBody
    public List<Evaluate> findEvaluates(@PathVariable("gid") Long goodsId){
        logger.info("请求评论");
        return goodsService.findEvaluates(goodsId);

    }


}
