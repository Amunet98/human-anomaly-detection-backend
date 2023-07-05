const express = require('express');
const app = express();
const server = require('http').Server(app);
const { PrismaClient } = require('@prisma/client');

/// making new prismaclient  object.
const prisma = new PrismaClient()

///configuring socket 
const io = require('socket.io')(server, {
	cors: {
		origin: "*"
	},
	transports: ["websocket", "polling"] 
});


//socket server listening  
io.on('connection', client => {
	console.log('connection established')
	client.on('data', data => {  
		var frame =  Buffer.from(data, 'base64').toString()
		io.emit('frame',frame)
		});
	client.on('detected',(data)=>{
		// dataBuffer = Buffer.from(data,'ascii').toString()
		// var splitted = dataBuffer.split('_')
		console.log(data)
		io.emit('detected',data)
	})
	client.on('disconnect', () => {
		console.log('client disconnected');
	});
  });
  


// Creating get request simple route
app.get('/', (req, res) => {
	res.send('Detection system')
});

//get Category route+controller
app.get('/category', async (req, res) => {
	const output = await prisma.category.findMany();
	res.send(output);
	res.status(200);
});


//get items route+controller
//pass category id 
app.get('/item/:id', async (req, res) => {
	try {
		const categoryId = req.params.id
		if (categoryId == null || undefined) {
			res.status(500);
			res.send('please pass category id in request params !');
		}
		const output = await prisma.items.findMany({
			where: {
				category_id: parseInt(categoryId)
			}
		});
		res.send(output);
		res.status(200);
	} catch (error) {
		console.log(error);
		res.status(500);
		res.send('internal Server error')

	}
});


//get items route+controller
//pass item id
app.get('/item/classes/:id', async (req, res) => {
	try {
		const itemId = req.params.id
		if (itemId == null || undefined) {
			res.status(500);
			res.send('please pass item id in request params !');
		}
		const output = await prisma.item_class_assign.findMany({
			where: {
				item_id: parseInt(itemId)
			}
		});
		res.send(output);
		res.status(200);
	} catch (error) {
		console.log(error);
		res.status(500);
		res.send('internal Server error')
	}
});


app.get('/detected', async (req, res) => {
	try {
		const output = await prisma.raw_data.findMany({
			orderBy: {
				time: 'desc'
			},
		});
		res.send(output);
		res.status(200);
	} catch (error) {
		console.log(error);
		res.status(500);
		res.send('internal Server error')
	}
});


// app.get('/', async (req, res) => {
// 	try {
// 		const itemId = req.params.id
// 		if (itemId == null || undefined) {
// 			res.status(500);
// 			res.send('please pass item id in request params !');
// 		}
// 		const output = await prisma.item_class_assign.findMany({
// 			where: {
// 				item_id: parseInt(itemId)
// 			}

// 		});
// 		res.send(output);
// 		res.status(200);
// 	} catch (error) {
// 		console.log(error);
// 		res.status(500);
// 		res.send('internal Server error')
// 	}
// });





// app.get('/check', (req, res) => {
// 	axios({
// 		method: "POST",
// 		url: "https://classify.roboflow.com/numbers-e7tsb/2",
// 		params: {
// 			api_key: "o638K18mmUlsYpQHLmn7"
// 		},
// 		data: image,
// 		headers: {
// 			"Content-Type": "application/x-www-form-urlencoded"
// 		}
// 	})
// 		.then(function (response) {
// 			console.log(response.data);
// 			res.send(response.data);
// 			res.statusCode(200);
// 		})
// 		.catch(function (error) {
// 			console.log(error.message);
// 			res.errored(error.message);
// 			res.statusCode(500);
// 		});
// 	res.sendFile(path.join(__dirname, 'index.html'));
// });

// Using setInterval to read the image every one second.

server.listen(5000);
