#!/usr/bin/env python

import re
import numpy as np
import sys

"""
Geqdsk object to read data from using techniques from Ben Dudson

Nick Walkden, May 2015 

"""
try:
  basestring
except NameError:
  basestring = str
  
def file_numbers(ingf):
	""" 
	Generator to read numbers in a file, originally written by Ben Dudson
	"""
	toklist = []
	while True:
		line = ingf.readline()
		if not line: break
		line = line.replace("NaN","-0.00000e0")
		pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?'	#regular expression to find numbers
		toklist = re.findall(pattern,line)
		for tok in toklist:
			yield tok

def get_next(obj):
    pyVer = sys.version_info[0]
    if pyVer == 2:
        return obj.next()
    else:
        return next(obj)
    

class Geqdsk:   
	def __init__(self,filename=None):
		self.data = {}
		self.flags = {'loaded' : False }

		if filename != None:
			self.read(filename)
								
					
	def read(self,filename):
		""" 
		Read in data 
		"""
		
		if isinstance(filename, basestring):
			self._filename = filename	#filename is a string, so treat as filename
			self._file = open(filename)
		else:
			#assume filename is an object
			self._file = filename
			self._filename = str(filename)
		

		#First line should be case, id number and dimensions
		line = self._file.readline()
		if not line:
			raise IOError("ERROR: Cannot read from file"+self._filename)			
		
		conts = line.split() 	#split by white space
		self.data['nw'] = int(conts[-2])	
		self.data['nh'] = int(conts[-1])
		self.data['idum'] = int(conts[-3])

		self.flags['case'] = conts[0:-4]
	
		#Now use generator to read numbers
		token = file_numbers(self._file)
		
		float_keys = [
		'rdim','zdim','rcentr','rleft','zmid',
		'rmaxis','zmaxis','simag','sibry','bcentr',
		'current','simag','xdum','rmaxis','xdum',
		'zmaxis','xdum','sibry','xdum','xdum']
		
		#read in all floats
		for key in float_keys:		              			
			self.data[key] = float(get_next(token))
		
		#Now read arrays
		def read_1d(n):
			data = np.zeros([n])
			for i in np.arange(n):
				data[i] = float(get_next(token))
			return data

		def read_2d(nx,ny):
			data = np.zeros([nx,ny])
			for i in np.arange(nx):
				data[i,:] = read_1d(ny)
			return data

		

		
		self.data['fpol'] = read_1d(self.data['nw'])
		self.data['pres'] = read_1d(self.data['nw'])
		self.data['ffprime'] = read_1d(self.data['nw'])
		self.data['pprime'] = read_1d(self.data['nw'])
		self.data['psirz'] = read_2d(self.data['nh'],self.data['nw'])
		self.data['qpsi'] = read_1d(self.data['nw'])
	
		#Now deal with boundaries
		self.data['nbbbs'] = int(get_next(token))
		self.data['limitr'] = int(get_next(token))

		def read_bndy(nb,nl):
			if nb > 0:			
				rb = np.zeros(nb)
				zb = np.zeros(nb)
				for i in np.arange(nb):
					rb[i] = float(get_next(token))
					zb[i] = float(get_next(token))
			else:
				rb = [0]
				zb = [0]
		
			if nl > 0:
				rl = np.zeros(nl)
				zl = np.zeros(nl)
				for i in np.arange(nl):
					rl[i] = float(get_next(token))
					zl[i] = float(get_next(token))
			else:
				rl = [0]
				zl = [0]

			return rb,zb,rl,zl


		self.data['rbbbs'],self.data['zbbbs'],self.data['rlim'],self.data['zlim'] = read_bndy(self.data['nbbbs'],self.data['limitr'])
		
		self.flags['loaded'] = True
		

	def dump(self,filename):
		import time
		from itertools import cycle
		cnt = cycle([0,1,2,3,4])
		def write_number(file,number,counter):
			if number < 0:
				seperator = "-"
				number = np.abs(number)
			else:
				seperator = " "
			if get_next(counter) == 4:
				last = "\n"
			else:
				last = ""
			
			string = '%.10E'%number
			#mant,exp = string.split('E')
			file.write(seperator+string+last)

		def write_1d(file,array,counter):
			for num in array:
				write_number(file,num,counter)

		def write_2d(file,array,counter):
			nx = array.shape[0]
			for i in np.arange(nx):
				write_1d(file,array[i],counter)
		
		def write_bndry(file,R,Z,counter):
			for i in np.arange(len(list(R))):
				write_number(file,R[i],counter)
				write_number(file,Z[i],counter)
			file.write("\n")
		
		with open(filename,'w') as file:
			line = " pyEquilibrium "+time.strftime("%d/%m/%Y")+" # 0 0 "+str(self.data['nw'])+" "+str(self.data['nh'])+"\n"
			file.write(line)

			float_keys = [
			'rdim','zdim','rcentr','rleft','zmid',
 			'rmaxis','zmaxis','simag','sibry','bcentr',
			'current','simag','xdum','rmaxis','xdum',
			'zmaxis','xdum','sibry','xdum','xdum']
			for key in float_keys:
				write_number(file,self.data[key],cnt)
			write_1d(file,self.data['fpol'],cnt)
			write_1d(file,self.data['pres'],cnt)
			write_1d(file,self.data['ffprime'],cnt)
			write_1d(file,self.data['pprime'],cnt)
			write_2d(file,self.data['psirz'],cnt)
			#No qpsi in eq object for now
			write_1d(file,np.zeros(self.data['nw']),cnt)	
			file.write("\n"+str(len(list(self.data['rbbbs'])))+"\t"+str(len(list(self.data['rlim'])))+"\n")
			write_bndry(file,self.data['rbbbs'],self.data['zbbbs'],cnt)
			write_bndry(file,self.data['rlim'],self.data['zlim'],cnt)


	def __getitem__(self,var):
		if self.flags['loaded']:
			return self.data[var]
		else:
			print("\nERROR: No gfile loaded")
			

	def get(self,var):
		if self.flags['loaded']:
			return self.data[var]
		else:
			print("\nERROR: No gfile loaded")
	def set(self,key,val):
		self.data[key] = val

