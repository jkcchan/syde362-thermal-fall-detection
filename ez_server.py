import web
import os

def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the latest image file in the given directory
    """
    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime) 

#---- server stuff
urls = (
    '/', 'index'
)
class index:
    def GET(self):
    	latest_image_path = get_latest_image('.')
        path = latest_image_path.split('/')[-1]
        cType = {
            "png":"images/png",
            "jpg":"images/jpeg",
            "gif":"images/gif",
            "ico":"images/x-icon" }
        if path in os.listdir('.'):  # Security
            web.header("Content-Type", cType['jpg']) # Set the Header
            return open('%s'%path,"rb").read() # Notice 'rb' for reading images
        else:
            raise web.notfound()

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()