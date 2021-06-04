
# data functions
def word2int_label(label):
    """
    transform word class to int_label (0~800)
    """
    word2int_label_dict = dict(zip(word_classes, range(len(word_classes))))
    return word2int_label_dict[label]

def int_label2word(int_label):
    """
    transform integer label to word class
    """
    int_label2word_dict = dict(zip(range(len(word_classes)), word_classes))
    return int_label2word_dict[int_label]

class ImageReader:
    @classmethod
    def read_image_pil(cls, path):
        """
        faster than cv2 7s vs 17 s
        """
        img = Image.open(path)
        return img

    @classmethod
    def read_image_cv2(cls, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @classmethod
    def show_image_and_label(cls, path, label, show_path=False):
        print(label, '\n')
        if show_path:
            print(path, '\n')
            
        image = cls.read_image_cv2(path)
        plt.imshow(image)
        plt.show()



class DataHandler:
    @classmethod
    def load_data(cls, file_paths, way='cv2'):
        data = []
        if way == 'cv2':
            data = [[ImageReader.read_image_cv2(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]                                   
        else:
            data = [[ImageReader.read_image_pil(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]
        images, labels = zip(*data)
        return images, labels

    # load data multiprocess
    @classmethod
    def load_data_mp(cls, file_paths):
        """
        normal: 
            cv2 1hr
            pil 23min
        multiprocessing:
            cv2 2min37s
            pil 3min50s
        """

        def worker(path, return_list, i):
            """worker function"""
            try:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ValueError:
                print(path)
            
            try:
                label = re.search('[\u4e00-\u9fa5]{1}', path).group(0)
            except:
                label = 'null'            
            return_list.append([img, label])
            if (i+1)%1000 == 0:
                print(f'{i+1} pictures loaded')

        manager = multiprocessing.Manager()
        return_list = manager.list()
        jobs = []
        print(f'total images: {len(file_paths)}\nstart loading...')
        for i, path in enumerate(file_paths):
            p = multiprocessing.Process(target=worker, args=(path, return_list, i))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        images, labels = zip(*return_list)
        return images, labels

class FileHandler:
    @classmethod
    def tar_file(cls, file_url):
        start = time.time()
        with tarfile.open(file_url) as file:
            file.extractall(os.path.dirname(file_url),)
            print(f'done! time: {time.time() - start}')

    @classmethod
    def copyfolder(cls, src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    @classmethod
    def save_path_and_label_as_txt(cls, target_path, paths):
        with open(target_path, 'w') as out_file:
            for path in paths:
                label = re.search("[\u4e00-\u9fa5]{1}", path).group(0)
                label = word2int_label(label)
                out_file.write(f'{path} {label}\n')

    @classmethod
    def read_path_and_label_from_txt(cls, target_path):
        with open(target_path) as in_file:
            lines = in_file.readlines()
            paths, labels = zip(*[[line.split(' ')[0], line.split(' ')[1].rstrip()] for line in lines])
        return paths, labels

class GpuHandler:
    @classmethod
    def see_gpu_usage_nvidia(cls):
        !nvidia-smi

    @classmethod
    def see_gpu_memory_objects(cls):
        import gc
        for obj in gc.get_objects():
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.__class__.__name__)