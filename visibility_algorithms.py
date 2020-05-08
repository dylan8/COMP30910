# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  December 2018
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import numpy as np

# ------------------------------------------------------------------------------
# NATURAL VISIBILITY GRAPH ( NVG )
# ------------------------------------------------------------------------------
# a. BASIC IMPLEMENTATION
# --------------------------
def nvg(series, timeLine):
    # series is the data vector to be transformed
    # Get length of input series
    L = len(series)
    # timeLine is the vector containing the time stamps
    #if timeLine == None: timeLine = range(L)


    # initialise output
    all_visible = []


    for i in range(L-1):
        node_visible = []
        ya = float(series[i])
        ta = timeLine[i]

        for j in range(i+1,L):
            yb = float(series[j])
            tb = timeLine[j]

            yc = series[i+1:j]
            tc = timeLine[i+1:j]

            if all( yc[k] < (ya + (yb - ya)*(tc[k] - ta)/(tb-ta)) for k in range(len(yc)) ):
                node_visible.append(tb)

        if len(node_visible)>0 : all_visible.append([ta, node_visible])

    return all_visible

# b. DIVIDE & CONQUER <---------------------- QUICKEST NVG IMPLEMENTATION
# --------------------------
def nvg_dc(series, timeLine, left, right, all_visible = None):

    #if timeLine == None : timeLine = range(len(series))
    if all_visible == None : all_visible = []
    
    if left==right:
        left+1

    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        k = series[left:right].index(max(series[left:right])) + left
    
    

        # check if k can see each node of series[left...right]
        for i in range(left,right):
    
            if i != k :
                a = min(i,k)
                b = max(i,k)

                ya = float(series[a])
                ta = timeLine[a]
                yb = float(series[b])
                tb = timeLine[b]
                yc = series[a+1:b]
                tc = timeLine[a+1:b]

                if all( yc[j] < (ya + (yb - ya)*(tc[j] - ta)/(tb-ta)) for j in range(len(yc)) ):
                    node_visible.append(timeLine[i])

        if len(node_visible) > 0 : all_visible.append([timeLine[k], node_visible])

        nvg_dc(series,timeLine,left, k, all_visible = all_visible)
        nvg_dc(series,timeLine, k+1, right, all_visible = all_visible)

    return all_visible

# a. NUMPY ORIGINAL IMPLEMENTATION
# --------------------------------
def nvg_np(series, timeLine):
    # !! SERIES ARE A NUMPY ARRAY HERE and SO IS TIMELINE !!
    # series is the data vector to be transformed
    # Get length of input series
    L = len(series)
    # timeLine is the vector containing the time stamps
    #if timeLine == None: timeLine = np.arange(L)

    # initialise output
    all_visible = []

    for i in np.arange(L-1):
        node_visible = []
        ya = float(series[i])
        ta = timeLine[i]

        for j in np.arange(i+1,L):
            yb = float(series[j])
            tb = timeLine[j]

            yc = series[i+1:j]
            tc = timeLine[i+1:j]


            if np.all( yc < (ya + (yb - ya)*(tc - ta)/(tb-ta))):
                node_visible.append(tb)

        if len(node_visible)>0 : all_visible.append([ta, node_visible])

    return all_visible


# b. NUMPY DIVIDE & CONQUER
# --------------------------
# !!!! SERIES IS A NUMPY ARRAY HERE AND SO IS TIMELINE !!!!
def nvg_dc_np(series, timeLine,  left, right, all_visible = None):

    if all_visible == None : all_visible = []

    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        k = np.argmax(series[left:right]) + left

        # check if k can see each node of series[left...right]
        for i in np.arange(left,right):
            if i != k :
                a = min(i,k)
                b = max(i,k)

                ya = series[a]
                ta = timeLine[a]
                yb = series[b]
                tb = timeLine[b]
                yc = series[a+1:b]
                tc = timeLine[a+1:b]

                if np.all( yc < (ya + (yb - ya)*(tc - ta)/(tb-ta))):
                    node_visible.append(timeLine[i])

        if len(node_visible) > 0 : all_visible.append([timeLine[k], node_visible])

        nvg_dc_np(series,timeLine, left, k, all_visible = all_visible)
        nvg_dc_np(series,timeLine, k+1, right, all_visible = all_visible)

    return all_visible


# ------------------------------------------------------------------------------
# HORIZONTAL VISIBILITY GRAPH ( HVG )
# ------------------------------------------------------------------------------

# a. ORIGINAL IMPLEMENTATION
# --------------------------
def hvg(series, timeLine):
    # series is the data vector to be transformed
    #if timeLine == None: timeLine = range(len(series))
    # Get length of input series
    L = len(series)
    # initialise output
    all_visible = []

    for i in range(L-1):
        node_visible = []
        ya = series[i]
        ta = timeLine[i]
        for j in range(i+1,L):

            yb = series[j]
            tb = timeLine[j]

            yc = series[i+1:j]
            tc = timeLine[i+1:j]

            if all( yc[k] < min(ya,yb) for k in range(len(yc)) ):
                node_visible.append(tb)
            elif all( yc[k] >= max(ya,yb) for k in range(len(yc)) ):
                break

        if len(node_visible)>0 : all_visible.append([ta, node_visible])

    return all_visible


# b. DIVIDE & CONQUER HVG
# --------------------------
def hvg_dc(series,timeLine, left, right, all_visible = None):

    if all_visible == None : all_visible = []

    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        k = series[left:right].index(max(series[left:right])) + left
        # check if k can see each node of series[left...right]

        for i in range(left,right):
            if i != k :
                a = min(i,k)
                b = max(i,k)

                ya = series[a]
                ta = timeLine[a]
                yb = series[b]
                tb = timeLine[b]
                yc = series[a+1:b]
                tc = timeLine[a+1:b]

                if all( yc[k] < min(ya,yb) for k in range(len(yc)) ):
                    node_visible.append(timeLine[i])
                elif all( yc[k] >= max(ya,yb) for k in range(len(yc)) ):
                    break

        if len(node_visible) > 0 : all_visible.append([timeLine[k], node_visible])

        hvg_dc(series,timeLine, left, k, all_visible = all_visible)
        hvg_dc(series,timeLine, k+1, right, all_visible = all_visible)

    return all_visible
