import tools
import os
import shutil

def filerCar_less_than(filepath,lowerbound=50,upperbound=200):
    """
    过滤掉文件数目小于下界的文件夹，从文件数量超过上界的文件夹中随机抽取上界个文件
    :param filepath: 文件夹位置
    :param lowerbound: 文件数量下界
    :param upperbound: 文件数量上界
    :return:
    """
    imagedirs = tools.getPicsDrilists(filepath)
    filelist = []
    totalnum = 0
    for dir in imagedirs:
        tmplist = os.listdir(dir)
        num = len(tmplist)
        if num <= lowerbound: continue
        if num >= upperbound:
            tmplist = tmplist[:upperbound]

        for img in tmplist:
            filelist.append(os.path.join(dir,img))
        totalnum+=1
    print(totalnum)
    return filelist

def copy_img(imglist,distroot):
    '''
    :param imglist: 待复制的文件列表
    :param distroot: 目标文件位置
    :return:
    '''
    for imgpath in imglist:
        spepath = imgpath.split(os.sep)
        maker,model,year = spepath[-4],spepath[-3],spepath[-2]
        distpath = os.path.join(distroot,maker,model,year)
        if not os.path.exists(distpath):
            os.makedirs(distpath)
        shutil.copy(imgpath,distpath)

if __name__ == '__main__':
    # get_all_label(r"C:\Users\caopan.58GANJI-CORP\data\pic_after_filer")
    filelist = filerCar_less_than(r"C:\Users\caopan.58GANJI-CORP\data\pic")
    # print(len(filelist))
    copy_img(filelist,distroot=r"C:\Users\caopan.58GANJI-CORP\data\pic_after_filer")