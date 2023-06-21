import pandas as pd
import numpy as np
from glob import glob
import cv2 as cv
import subprocess


def make_results_csv(path_to_video):
  video = path_to_video + '/results.txt'
  results = pd.read_csv(video, header = None)
  results.columns = ['frame','id','boxx','boxy','boxw','boxh','class']
  results['centx'] = results['boxx'] + results['boxw']/2
  results['centy'] = results['boxy'] + results['boxh']/2
  results['diagofbox'] = np.sqrt(results['boxw']**2 + results['boxh']**2)
  results['m'] = float('nan')
  results['distinpixels'] = float('nan')
  results['nearedges'] = 0

  ind = (results['diagofbox'] >= 75) & (results['diagofbox'] <= 450)
  results = results[ind]
  results = results.reset_index(drop = True)

  pd.options.mode.chained_assignment = None  # default='warn'

  for i in results['id'].unique():
    a = results.groupby(['id']).get_group(i).diff()
    m = (a['centy'] / (a['centx']))
    dip = (np.sqrt(a['centx']**2 + a['centy']**2)[a.index]/a['frame'])

    nearedges = (results['boxx'][a.index] < 5) | (results['boxx'][a.index] + results['boxw'][a.index] > 1275) | (results['boxy'][a.index] < 5) | (results['boxy'][a.index] + results['boxh'][a.index] > 715)
    lowmov = (abs(a['centx']) < 0.025 * results['diagofbox'][a.index]) & (abs(a['centy']) < 0.025 * results['diagofbox'][a.index])

    m[lowmov] = float('nan')
    dip[lowmov] = 0

    results['m'][a.index] = m.rolling(window=3, center = True, min_periods = 1).mean()
    results['distinpixels'][a.index] = dip.rolling(window=3, center = True, min_periods = 1).mean()

    results['m'][a.index[(a['frame'] > 3) | (m.isna())]] = float('nan')
    results['distinpixels'][a.index[(a['frame'] > 3) | (dip.isna())]] = float('nan')
    results['nearedges'][a.index[nearedges]] = 1

  results['angle']  = np.degrees(np.arctan(results['m']))

  results['maskpointx'] = float('nan')
  results['maskpointy'] = float('nan')
  results['pixels'] = float('nan')
  results['distperpixel'] = float('nan')
  results['orientation'] = float('nan')
  results = results[results['nearedges'] == 0]
  results = results.reset_index(drop = True)
  results = results.drop('nearedges', axis = 1)

  for i in range(len(results)):
    try:
      maskfile = pd.read_csv(video.split("results.txt")[0]+'masks/'+str(results['frame'][i])+','+str(results['id'][i])+'.csv', header = None)
    except:
      print("Empty",video.split("results.txt")[0]+'masks/'+str(results['frame'][i])+','+str(results['id'][i]))
    maskfile.columns = ['y','x']
    ncols = round(results['diagofbox'][i]*0.05)

    x = maskfile['x']
    y = maskfile['y']

    results['orientation'][i] = np.cov(y, x, bias = True)[0][1]/( sum(i*i for i in x)/len(x) - np.mean(x)**2 )

    if results['class'][i] in [2,4]:
      results['pixels'][i] = results['boxh'][i]
      results['distperpixel'][i] = 1.12/results['pixels'][i]
      continue

    if results['class'][i] == 3:
      vehiclelength = 3.845
      vehiclewidth = 1.735
    if results['class'][i] == 6:
      vehiclelength = 12
      vehiclewidth = 2.55
    if results['class'][i] == 8:
      vehiclelength = 4
      vehiclewidth = 2.03

    if ((abs(results['angle'][i]) < 95) & (abs(results['angle'][i]) > 85)) | (abs(results['angle'][i]) < 5):
      results['pixels'][i] = results['boxw'][i]
      results['distperpixel'][i] = vehiclewidth/results['pixels'][i]
      continue

    if results['angle'][i] > 5:
      ind = maskfile['x'] < min(maskfile['x']) + ncols
      maxid = maskfile['y'][ind].idxmax()
      results['maskpointx'][i] = maskfile['x'][maxid]
      results['maskpointy'][i] = maskfile['y'][maxid]
      results['pixels'][i] = (max(results['boxy'][i] + results['boxh'][i], max(maskfile['y'])) - (results['maskpointy'][i] - results['m'][i] * results['maskpointx'][i]))/results['m'][i] - min(results['boxx'][i], min(maskfile['x']))
      results['distperpixel'][i] = (vehiclelength * np.cos(np.deg2rad(results['angle'][i])))/results['pixels'][i]
      continue

    if results['angle'][i] < -5:
      ind = maskfile['x'] > max(maskfile['x']) - ncols
      maxid = maskfile['y'][ind].idxmax()
      results['maskpointx'][i] = maskfile['x'][maxid]
      results['maskpointy'][i] = maskfile['y'][maxid]

      results['pixels'][i] = max(results['boxx'][i] + results['boxw'][i], max(maskfile['x'])) - (max(results['boxy'][i] + results['boxh'][i], max(maskfile['y'])) - (results['maskpointy'][i] - results['m'][i] * results['maskpointx'][i]))/results['m'][i]
      results['distperpixel'][i] = (vehiclelength * np.cos(np.deg2rad(results['angle'][i])))/results['pixels'][i]

  results['speed_m/s'] = results['distinpixels'] * results['distperpixel'] * 10
  results['speed_m/s'][results['speed_m/s'].isna()] = 0
  results['speed_m/s'][results['distinpixels'].isna()] = float('nan')

  results.to_csv(path_to_video + '/results.csv', index=False)

  
def make_rules(path):
  results = pd.read_csv(path+'/results.csv')

  rules = pd.DataFrame(columns = ['frame', 'v1_id', 'v1_class', 'v1_maxspeed', 'v1_avgspeed', 'v1_samedir', 'v1_coef', 'v1_coef2',
                                           'v2_id', 'v2_class', 'v2_maxspeed', 'v2_avgspeed', 'v2_samedir', 'v2_coef', 'v2_coef2',
                                  'intersection_angle'])

  for frame in range(8,max(results['frame'])):
    #print('frame number:' + str(frame))
    ids = sorted(results['id'][(results['frame'] <= frame) & (results['frame'] > frame - 7)].unique().tolist())
    info = []

    for id in ids:
      ind = (results['id'] == id) & (results['frame'] <= frame) & (results['frame'] > frame - 7)
      x = np.vstack([np.ones(sum(ind)),results['centx'][ind].to_numpy()])
      y = results['centy'][ind].to_numpy()
      vclass = results['class'][ind].iloc[-1]
      diagofbox = results['diagofbox'][ind].iloc[-1]

      deltax = max(x[1]) - min(x[1])
      deltay = max(y) - min(y)

      vertex_tl = np.sqrt( (results['boxx'][ind].iloc[0] - results['boxx'][ind].iloc[-1])**2 + (results['boxy'][ind].iloc[0] - results['boxy'][ind].iloc[-1])**2 )
      vertex_bl = np.sqrt( (results['boxx'][ind].iloc[0] - results['boxx'][ind].iloc[-1])**2 + (results['boxy'][ind].iloc[0] + results['boxh'][ind].iloc[0] - results['boxy'][ind].iloc[-1] - results['boxh'][ind].iloc[-1])**2 )
      vertex_tr = np.sqrt( (results['boxx'][ind].iloc[0] + results['boxw'][ind].iloc[0] - results['boxx'][ind].iloc[-1] - results['boxw'][ind].iloc[-1])**2 + (results['boxy'][ind].iloc[0] - results['boxy'][ind].iloc[-1])**2 )
      vertex_br = np.sqrt( (results['boxx'][ind].iloc[0] + results['boxw'][ind].iloc[0] - results['boxx'][ind].iloc[-1] - results['boxw'][ind].iloc[-1])**2 + (results['boxy'][ind].iloc[0] + results['boxh'][ind].iloc[-1] - results['boxy'][ind].iloc[-1] - results['boxh'][ind].iloc[0])**2 )

      boxinc = (vertex_tl < 7) | (vertex_bl < 7) | (vertex_tr < 7) | (vertex_br < 7)

      maxspeed = np.nanmax(results['speed_m/s'][ind])
      if np.isnan(maxspeed): continue
      avgspeed = np.nanmean(results['speed_m/s'][ind])

      #SDspeed = np.std(results['speed_m/s'][ind])

      #m1 = np.array(results['orientation'][ind])[0]
      #m2 = np.array(results['orientation'][ind])[1:]
      #change_angle = change_angle = np.append(np.degrees(np.arctan((m1 - m2)/(1 + m1 * m2))), 0)
      #SDorientatiovideon = np.std(change_angle)

      if ((deltax < 0.07*diagofbox) & (deltay < 0.07*diagofbox)) | (boxinc) | (np.linalg.det(x@np.transpose(x)) == 0) | (maxspeed == 0):
        beta = ['car is still', 'car is still']
        direction = 'car is still'
      else:
        beta = ((np.linalg.inv(x@np.transpose(x))) @ (x@np.transpose(y))).tolist()
        if deltax > deltay:
          if x[1][0] < x[1][-1]: direction = 'East'
          else: direction = 'West'
        else:
          if y[0] < y[-1]: direction = 'South'
          else: direction = 'North'
      idinfo = [id, vclass, direction, maxspeed] + beta + [x[1][-1], y[-1], diagofbox, avgspeed]#, SDspeed, SDorientation]
      info.append(idinfo)

    for i in range(len(info)):
      
      for j in range(i + 1, len(info)):
        if (info[i][4] == 'car is still') | (info[j][4] == 'car is still'):
          rules.loc[len(rules)] = [frame] + info[i][0:2] + [info[i][3], info[i][9], -1, -1, -1] + info[j][0:2] + [info[j][3], info[j][9], -1, -1, -1] + [-1]
        else:
          intersectx = (info[j][4] - info[i][4])/(info[i][5] - info[j][5])
          intersecty = info[i][4] + info[i][5] * intersectx
          distfromi = np.sqrt((intersectx - info[i][6])**2 + (intersecty - info[i][7])**2)
          distfromj = np.sqrt((intersectx - info[j][6])**2 + (intersecty - info[j][7])**2)

          if intersectx > info[i][6]: dirxfromi = 'East'
          else: dirxfromi = 'West'
          if intersecty > info[i][7]: diryfromi = 'South'
          else: diryfromi = 'North'
          if intersectx > info[j][6]: dirxfromj = 'East'
          else: dirxfromj = 'West'
          if intersecty > info[j][7]: diryfromj = 'South'
          else: diryfromj = 'North'

          if (info[i][2] == dirxfromi) | (info[i][2] == diryfromi) : samediri = 1
          else: samediri = 0
          if (info[j][2] == dirxfromj) | (info[j][2] == diryfromj) : samedirj = 1
          else : samedirj = 0

          coefi = distfromi/info[i][8]
          coefj = distfromj/info[j][8]

          coef2i = coefi/info[i][9]
          coef2j = coefj/info[j][9]

          b1i = info[i][5]
          b1j = info[j][5]

          if (info[i][2] == 'North'):
            xi = intersectx + 10/b1i
            yi = intersecty + 10
          if (info[i][2] == 'South'):
            xi = intersectx - 10/b1i
            yi = intersecty - 10
          if (info[i][2] == 'West'):
            xi = intersectx + 10
            yi = intersecty + 10*b1i
          if (info[i][2] == 'East'):
            xi = intersectx - 10
            yi = intersecty - 10*b1i

          if (info[j][2] == 'North'):
            xj = intersectx + 10/b1j
            yj = intersecty + 10
          if (info[j][2] == 'South'):
            xj = intersectx - 10/b1j
            yj = intersecty - 10
          if (info[j][2] == 'West'):
            xj = intersectx + 10
            yj = intersecty + 10*b1j
          if (info[j][2] == 'East'):
            xj = intersectx - 10
            yj = intersecty - 10*b1j

          intersection_angle = abs(np.degrees(np.arctan2(yj - intersecty, xj - intersectx) - np.arctan2(yi - intersecty, xi - intersectx))) # https://stackoverflow.com/a/31334882
          intersection_angle = min(intersection_angle, 360 - intersection_angle)

          rules.loc[len(rules)] = [frame] + info[i][0:2] + [info[i][3], info[i][9], samediri, coefi, coef2i] + info[j][0:2] + [info[j][3], info[j][9], samedirj, coefj, coef2j] + [intersection_angle]

  rules.to_csv( path + '/trajectory.csv', index=False)


def crash_video(path):
    r = pd.read_csv(path+ '/trajectory.csv')

    coef_condition = (r['v1_coef'] > 0.229) & (r['v1_coef'] < 0.63) & (r['v2_coef'] > 0.229) & (r['v2_coef'] < 0.63) & (r['v1_coef2'] > 0) & (r['v1_coef2'] < 1) & (r['v2_coef2'] > 0) & (r['v2_coef2'] < 1)
    dir_condition = (r['v1_samedir'] == 0) & (r['v2_samedir'] == 0)
    angle_condition = (r['intersection_angle'] > 15)

    fr = r[(coef_condition) & ~(dir_condition)] # & (angle_condition)]
    fr = fr.reset_index()
    results = pd.read_csv(path + "/results.csv")
    crash_list = []
    crashes = {}
    for i in range(len(fr)):

        v1_id = fr['v1_id'][i]
        v2_id = fr['v2_id'][i]
        frame = fr['frame'][i]

        ind1 = (results['id'] == v1_id) & (results['frame'] > frame - 11) & (results['frame'] < frame + 11)
        dist1 = results[['frame', 'centx', 'centy', 'diagofbox']][ind1]

        ind2 = (results['id'] == v2_id) & (results['frame'] > frame - 11) & (results['frame'] < frame + 11)
        dist2 = results[['frame', 'centx', 'centy', 'diagofbox']][ind2]

        dist = pd.merge(dist1, dist2, on = 'frame')
        dist['dist_coef'] = np.sqrt( (dist['centx_x'] - dist['centx_y'])**2 + (dist['centy_x'] - dist['centy_y'])**2 ) / dist['diagofbox_x'] / dist['diagofbox_y'] * 1000

        indpre = dist['frame'] <= frame
        pre = dist[['frame', 'dist_coef']][indpre]
        post = dist[['frame', 'dist_coef']][~indpre]

        if len(pre) > 2: pre_slope = np.cov(pre['dist_coef'], pre['frame'], bias = True)[0][1]/( sum(i*i for i in pre['frame'])/len(pre['frame']) - np.mean(pre['frame'])**2 )
        else: pre_slope = 0
        if len(post) > 2: post_slope = np.cov(post['dist_coef'], post['frame'], bias = True)[0][1]/( sum(i*i for i in post['frame'])/len(post['frame']) - np.mean(post['frame'])**2 )
        else: post_slope = 0

        change = abs(pre_slope - post_slope)

        if (pre_slope > 0.3): continue
        ind = (r['v1_id'] == v1_id) & (r['v2_id'] == v2_id) & (r['frame'] == frame)
        ia = int(r['intersection_angle'][ind])


        #print(frame, v1_id, v2_id, pre_slope, post_slope, change, ia)

        if ((change > 0.25) & (post_slope < 0.3)) | (ia > 45):
            #print(frame, v1_id, v2_id, pre_slope, post_slope, change, ia)
            ms1 = int(r['v1_maxspeed'][ind])
            ms2 = int(r['v2_maxspeed'][ind])
            print('A crash has occured between vehicles ' + str(v1_id) + ' and ' + str(v2_id) + ' on frame number ' + str(frame))
            try:
                crashes[(v1_id,v2_id)].append(frame)
                crash_list.append((v1_id,v2_id,frame))
            except:
                crashes[(v1_id,v2_id)] = [frame]
            print('_____')
    
    cap = cv.VideoCapture(path + "/results.avi")

    path_to_masks = path + "/masks/"
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(path + '/output.avi', fourcc, 10, (1280, 720))
    frame=-1
    while cap.isOpened():
        ret, img = cap.read()
        frame+=1
        if frame<2:
            continue
        ids = glob(f"{path_to_masks}{frame},*")
        r,g,b=102,204,255
        alpha=0.4
        seen=[]
        for id in ids: #102,204,255 rgb
            carid = int(id.split(",")[-1].strip('.csv'))
            mask = pd.read_csv(id,header=None)
            for cars in crashes:
                min_crash_frame = min(crashes[cars])
                max_crash_frame = max(crashes[cars])
                if (min_crash_frame-5<frame) & (carid in cars):
                    img[mask[0],mask[1],1] = img[mask[0],mask[1],1]*alpha + 153*(1-alpha)
                    img[mask[0],mask[1],2] = img[mask[0],mask[1],2]*(alpha) + 255*(1-alpha)
                    img[mask[0],mask[1],0] = b*(alpha)
                    seen.append((carid,frame))
                else:
                    if (carid,frame) in seen:
                        continue
                    img[mask[0],mask[1],1] = img[mask[0],mask[1],1]*(alpha) + g*(1-alpha)
                    img[mask[0],mask[1],2] = img[mask[0],mask[1],2]*(alpha) + r*(1-alpha)
                    img[mask[0],mask[1],0] = b*(alpha)
                if (min_crash_frame<=frame<=max_crash_frame) & (carid in cars):
                    img[mask[0],mask[1],1] = img[mask[0],mask[1],1]*alpha
                    img[mask[0],mask[1],2] = img[mask[0],mask[1],2]*(alpha) + 255*(1-alpha)
                    img[mask[0],mask[1],0] = b*(alpha)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        out.write(img)
        # cv2_imshow(img)
        if cv.waitKey(1) == ord('q'):
            break
        # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()




def detect_crash(video_path,absolute_output_folder_path):
    subprocess.run(["python", "deepsort.py", video_path, "--save_path", absolute_output_folder_path])
    make_results_csv(absolute_output_folder_path)
    make_rules(absolute_output_folder_path)
    crash_video(absolute_output_folder_path)
   