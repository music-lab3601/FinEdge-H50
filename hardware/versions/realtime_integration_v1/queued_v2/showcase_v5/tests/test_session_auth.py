from pathlib import Path
import http.cookiejar,json,sys,threading,unittest,urllib.request,urllib.error
U=Path(__file__).resolve().parents[1];sys.path.insert(0,str(U))
from dashboard import make_server

class EmptyApp:
    def state(self):return dict(schema='session-test')

class SessionAuthTests(unittest.TestCase):
    def test_live_and_review_services_share_a_browser_without_overwriting_auth(self):
        servers=[make_server(EmptyApp(),0),make_server(EmptyApp(),0)]
        threads=[threading.Thread(target=s.serve_forever,daemon=True) for s in servers]
        for t in threads:t.start()
        try:
            jar=http.cookiejar.CookieJar();op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
            urls=['http://127.0.0.1:'+str(s.server_port) for s in servers]
            for url in urls:op.open(url,timeout=3).read()
            for url in urls:self.assertEqual(json.load(op.open(url+'/api/state',timeout=3))['schema'],'session-test')
            self.assertEqual(len({cookie.name for cookie in jar}),2)
            # An unauthenticated request must remain denied after isolation.
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(urls[0]+'/api/state',timeout=3)
            self.assertEqual(error.exception.code,403)
        finally:
            for s in servers:s.shutdown();s.server_close()
            for t in threads:t.join()

if __name__=='__main__':unittest.main()
