# openstDB python Demo

	# coding:utf-8
	import requests
	
	
	payload = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586540',
	    "value": '29',
	    "tags": {
	        "host": "web01"
	    }
	}
	
	payload1 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586541',
	    "value": '30',
	    "tags": {
	        "host": "web01"
	    }
	}
	
	payload2 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586542',
	    "value": '29',
	    "tags": {
	        "host": "web02"
	    }
	}
	
	payload3 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586543',
	    "value": '23',
	    "tags": {
	        "host": "web01"
	    }
	}
	
	payload4 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586544',
	    "value": '23',
	    "tags": {
	        "host": "web02"
	    }
	}
	
	payload5 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586545',
	    "value": '33',
	    "tags": {
	        "host": "web01"
	    }
	}
	
	payload6 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586546',
	    "value": '22',
	    "tags": {
	        "host": "web02"
	    }
	}
	
	payload7 = {
	    "metric": "sys.cpu.data",
	    "timestamp": '1490586547',
	    "value": '23',
	    "tags": {
	        "host": "web02"
	    }
	}
	
	ls = [payload, payload1, payload2, payload3, payload4, payload5, payload6, payload7]
	
	
	def send_json(json):
	    s = requests.Session()
	    r = s.post("http://47.112.142.231:4242/api/put?details", json=json)
	    return r.text
	
	
	if __name__ == "__main__":
	    print(send_json(ls))