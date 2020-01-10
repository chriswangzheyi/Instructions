package com.wzy.springboot_anti_xss.utils;


import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletRequestWrapper;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.text.StringEscapeUtils;

public class XssAndSqlHttpServletRequestWrapper extends HttpServletRequestWrapper {

    private HttpServletRequest request;

    public XssAndSqlHttpServletRequestWrapper(HttpServletRequest request) {
        super(request);
        this.request = request;
    }

    /**
     * 假如有有html 代码是自己传来的  需要设定对应的name 不走StringEscapeUtils.escapeHtml4(value) 过滤
     */
    @Override
    public String getParameter(String name) {
        String value = request.getParameter(name);
        if (!StringUtils.isEmpty(value)) {
            value = StringEscapeUtils.escapeHtml4(value);
        }
        return value;
    }

    @Override
    public String[] getParameterValues(String name) {
        String[] parameterValues = super.getParameterValues(name);
        if (parameterValues == null) {
            return null;
        }
        for (int i = 0; i < parameterValues.length; i++) {
            String value = parameterValues[i];
            parameterValues[i] = StringEscapeUtils.escapeHtml4(value);
        }
        return parameterValues;
    }
}

