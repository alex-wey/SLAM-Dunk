def main:
	imgl = load(...)
	imgr = load(...)
	C = zeros(...)# current camera pose
	For i in range(n):
		#frame1
		Matchesl1, matchesr1= matchFeatures(imgl(i), imgr(i))
		Matchesl1, matchesr1  = RANSAC(Matchesl, matchesr1)
		3Dcoords1 = triangulate(Matchesl, matchesr1)
		#frame2
		Matchesl2, matchesr2 = matchFeatures(imgl(i+1), imgr(i+1))
		Matchesl2, matchesr2  = RANSAC(Matchesl2, matchesr2)
		3Dcoords2 = triangulate(Matches2, matchesr2)
		updateCameraPose(3Dcoord1, 3Dcoord2) #updates C
