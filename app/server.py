import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import json

export_file_url = 'https://drive.google.com/uc?export=download&id=1QSHgV_BeXhhVei4jTPiW-eEjnr23P97H'
export_file_name = 'CattleRec_Resnet18.pkl'


classes = ['Balwindr C2', 'Gursewk CL11', 'Gursewk CL12', 'Gursewk CL3', 'Gursewk CL4', 'Gursewk CL5', 'Gursewk Shamdoo CL 2', 
'Gursewk Shamdoo CL 3', 'Gursewk Shamdoo CL 4', 'Gursewk Shamdoo CL 8', 'Jaswnt CL 2', 'Jaswnt CL 3', 'Lovdp CL 1', 'mnpreet Cl 8', 
'mnpreet cl 10', 'mohan cl 1', 'mohan cl 10', 'mohan cl 11', 'mohan cl 12', 'mohan cl 13', 'mohan cl 14', 'mohan cl 15', 'mohan cl 16', 
'mohan cl 17', 'mohan cl 18', 'mohan cl 19', 'mohan cl 2', 'mohan cl 20', 'mohan cl 21', 'mohan cl 3', 'mohan cl 4', 'mohan cl 5', 'mohan cl 6', 'mohan cl 8', 'mohan cl 9', 'ramtej cl10', 'ramtej cl12', 'ramtej cl13', 'ramtej cl14', 'ramtej cl15', 'ramtej cl16', 'ramtej cl17', 'ramtej cl18','ramtej cl19', 'ramtej cl2', 'ramtej cl20', 'ramtej cl22', 'ramtej cl23', 'ramtej cl3', 'ramtej cl4', 'ramtej cl5', 'ramtej cl6', 'ramtej cl7', 'ramtej cl8','ramtej cl9']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

            

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    print(img_data)
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    print(learn.predict(img))
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
